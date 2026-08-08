import Mathlib
import Mathlib.MeasureTheory.Measure.Lebesgue.EqHaar
import Mathlib.Analysis.Convex.Measure

open MeasureTheory ProbabilityTheory Set Filter
open scoped Pointwise ENNReal Topology

namespace Bounty

abbrev I7 := ↥(Finset.range 7)

noncomputable def μ7 : Measure (I7 → ℝ) :=
  Measure.pi (fun _ : I7 => gaussianReal 0 1)

def ratioCone (n : ℕ) : Set (I7 → ℝ) :=
  {x | ∀ i j, |x i| ≤ (n : ℝ) * |x j|}

def noZero : Set (I7 → ℝ) :=
  {x | ∀ i, x i ≠ 0}

lemma ratioCone_isClosed (n : ℕ) : IsClosed (ratioCone n) := by
  change IsClosed {x : I7 → ℝ | ∀ i j, |x i| ≤ (n : ℝ) * |x j|}
  simp only [setOf_forall]
  exact isClosed_iInter fun i => isClosed_iInter fun j =>
    isClosed_le ((continuous_apply i).abs)
      (continuous_const.mul ((continuous_apply j).abs))

lemma ratioCone_measurable (n : ℕ) : MeasurableSet (ratioCone n) :=
  (ratioCone_isClosed n).measurableSet

lemma ratioCone_smul_mem (n : ℕ) {x : I7 → ℝ} (hx : x ∈ ratioCone n) (a : ℝ) :
    a • x ∈ ratioCone n := by
  intro i j
  simpa [Pi.smul_apply, smul_eq_mul, abs_mul, mul_assoc, mul_left_comm, mul_comm] using
    mul_le_mul_of_nonneg_left (hx i j) (abs_nonneg a)

lemma ratioCone_balanced (n : ℕ) : Balanced ℝ (ratioCone n) := by
  rw [balanced_iff_smul_mem]
  intro a _ x hx
  exact ratioCone_smul_mem n hx a

lemma ratioCone_mono : Monotone ratioCone := by
  intro n m hnm x hx i j
  exact (hx i j).trans <| by
    gcongr
    exact_mod_cast hnm

lemma ratioCone_eq_zero_of_coord_eq_zero (n : ℕ) {x : I7 → ℝ}
    (hx : x ∈ ratioCone n) {i : I7} (hi : x i = 0) : x = 0 := by
  funext j
  have h := hx j i
  rw [hi, abs_zero, mul_zero] at h
  exact abs_eq_zero.mp (le_antisymm h (abs_nonneg _))

lemma noZero_subset_iUnion_ratioCone : noZero ⊆ ⋃ n : ℕ, ratioCone n := by
  intro x hx
  obtain ⟨M, hM⟩ := Finite.exists_le (fun p : I7 × I7 => |x p.1| / |x p.2|)
  obtain ⟨n, hn⟩ := exists_nat_ge M
  refine mem_iUnion.2 ⟨n, ?_⟩
  intro i j
  have hj : 0 < |x j| := abs_pos.mpr (hx j)
  exact (div_le_iff₀ hj).mp ((hM (i, j)).trans hn)

lemma μ7_noZero : μ7 noZero = 1 := by
  letI : NoAtoms (gaussianReal 0 1) := noAtoms_gaussianReal (by norm_num)
  rw [show noZero = univ.pi (fun _ : I7 => ({0}ᶜ : Set ℝ)) by
    ext x
    simp [noZero]]
  simp [μ7, Measure.pi_pi]

lemma μ7_iUnion_ratioCone : μ7 (⋃ n : ℕ, ratioCone n) = 1 := by
  apply le_antisymm
  · exact measure_le_one
  · rw [← μ7_noZero]
    exact measure_mono noZero_subset_iUnion_ratioCone

lemma exists_ratioCone_large : ∃ n : ℕ, (0.99 : ℝ≥0∞) < μ7 (ratioCone n) := by
  have hlim : Tendsto (fun n : ℕ => μ7 (ratioCone n)) atTop (𝓝 1) := by
    simpa [Function.comp_def, μ7_iUnion_ratioCone] using
      (tendsto_measure_iUnion_atTop (μ := μ7) ratioCone_mono)
  have hev : ∀ᶠ n : ℕ in atTop, (0.99 : ℝ≥0∞) < μ7 (ratioCone n) :=
    hlim.eventually (Ioi_mem_nhds (by norm_num))
  exact (eventually_atTop.1 hev).imp fun n hn => ⟨n, hn n le_rfl⟩

end Bounty
