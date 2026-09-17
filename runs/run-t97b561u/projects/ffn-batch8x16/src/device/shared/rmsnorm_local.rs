//! QKV experiment: materialize HBM input on every projection slice, then
//! normalize locally. No relabel of a partly populated slice layout is used.
//! The H reduction is now per-slice, so f32 summation order differs from the
//! baseline's eight partial reductions. The unchanged RNGD tests are required.

use furiosa_opt_std::prelude::*;

use crate::axes::H;
use crate::device::layout::{Cluster, Replicated};
use crate::{Chip, EPS};

const H_F32: f32 = H::SIZE as f32;
const TILE: usize = 480;

pub(crate) fn normalize_from_hbm(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Replicated, m![H]> {
    // These are real HBM -> DM transfers. Both x and gamma use exactly the
    // Replicated mapping consumed by Q/K/V projection (Dummy256 x H).
    let x: DmTensor<bf16, Chip, Cluster, Replicated, m![H]> = x.to_dm(&mut ctx.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, Replicated, m![H]> =
        rms_weight.to_dm(&mut ctx.tdma);

    // Every slice reduces ALL H=3840 entries of its own x. Do not reduce
    // Dummy256: those are independent copies, not portions of the H sum.
    let mean_square: DmTensor<f32, Chip, Cluster, Replicated, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, Replicated, m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms_vrf: VrfTensor<f32, Chip, Cluster, Replicated, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    // Keep only 480 f32 weights (1920 bytes per slice) in one weight VRF
    // operand, instead of a full 3840-value weight operand. These eight
    // disjoint views cover H exactly once and retain the same slice mapping.
    let mut normalized: DmTensor<bf16, Chip, Cluster, Replicated, m![H]> = DmTensor::new();
    for tile in 0..H::SIZE / TILE {
        let weight_vrf: VrfTensor<f32, Chip, Cluster, Replicated, m![H % 480]> = ctx
            .sub
            .begin(weight.view().tile::<m![H / 480], 1, m![H / 480 = 1 # 8, H % 480]>(tile))
            .fetch::<m![H / 16 % 30], m![H % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![H / 8 % 60], m![H % 8]>()
            .to_vrf();
        ctx.main
            .begin(x.view().tile::<m![H / 480], 1, m![H / 480 = 1 # 8, H % 480]>(tile))
            .fetch::<m![H / 16 % 30], m![H % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![H / 8 % 60], m![H % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
            .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
            .vector_final()
            .cast::<bf16, m![H % 8 # 16]>()
            .commit_trim::<m![H % 8]>()
            .commit_view(normalized.view_mut().tile::<m![H / 480], 1, m![H / 480 = 1 #{!} 8, H % 480]>(tile));
    }
    normalized
}
