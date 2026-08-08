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
  have hnm' : (n : ℝ) ≤ (m : ℝ) := by exact_mod_cast hnm
  exact (hx i j).trans (mul_le_mul_of_nonneg_right hnm' (abs_nonneg _))

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
  have hcomp : gaussianReal 0 1 ({0}ᶜ : Set ℝ) = 1 := by
    rw [measure_compl (MeasurableSet.singleton 0)] <;> simp
  rw [show noZero = univ.pi (fun _ : I7 => ({0}ᶜ : Set ℝ)) by
    ext x
    simp [noZero]]
  rw [μ7, Measure.pi_pi]
  simp [hcomp]

lemma μ7_iUnion_ratioCone : μ7 (⋃ n : ℕ, ratioCone n) = 1 := by
  apply le_antisymm
  · calc
      μ7 (⋃ n : ℕ, ratioCone n) ≤ μ7 (univ : Set (I7 → ℝ)) :=
        measure_mono (subset_univ _)
      _ = 1 := measure_univ
  · rw [← μ7_noZero]
    exact measure_mono noZero_subset_iUnion_ratioCone

lemma exists_ratioCone_large : ∃ n : ℕ, (0.99 : ℝ≥0∞) < μ7 (ratioCone n) := by
  have hlim : Tendsto (fun n : ℕ => μ7 (ratioCone n)) atTop (𝓝 1) := by
    simpa [Function.comp_def, μ7_iUnion_ratioCone] using
      (tendsto_measure_iUnion_atTop (μ := μ7) ratioCone_mono)
  have h99 : (0.99 : ℝ≥0∞) < 1 := by
    change (99 / 100 : ℝ≥0∞) < 1
    norm_num
  have hev : ∀ᶠ n : ℕ in atTop, (0.99 : ℝ≥0∞) < μ7 (ratioCone n) :=
    hlim.eventually (Ioi_mem_nhds h99)
  rcases eventually_atTop.1 hev with ⟨n, hn⟩
  exact ⟨n, hn n le_rfl⟩

section ProductAbsoluteContinuity

variable {δ : Type*} {X : δ → Type*} [∀ i, MeasurableSpace (X i)]
variable {μ ν : ∀ i, Measure (X i)} [∀ i, SigmaFinite (ν i)]

lemma tprod_absolutelyContinuous (h : ∀ i, μ i ≪ ν i) (l : List δ) :
    Measure.tprod l μ ≪ Measure.tprod l ν := by
  induction l with
  | nil => exact Measure.AbsolutelyContinuous.rfl
  | cons i l ih =>
      rw [Measure.tprod_cons, Measure.tprod_cons]
      exact (h i).prod ih

variable [Fintype δ] [Encodable δ] [∀ i, SigmaFinite (μ i)]

lemma pi_absolutelyContinuous (h : ∀ i, μ i ≪ ν i) :
    Measure.pi μ ≪ Measure.pi ν := by
  classical
  rw [← Measure.pi'_eq_pi μ, ← Measure.pi'_eq_pi ν]
  unfold Measure.pi'
  exact (tprod_absolutelyContinuous h (Encodable.sortedUniv δ)).map
    (MeasurableEquiv.piMeasurableEquivTProd
      (Encodable.sortedUniv_nodup δ) (Encodable.mem_sortedUniv)).symm.measurable

end ProductAbsoluteContinuity

lemma μ7_absolutelyContinuous_volume : μ7 ≪ (volume : Measure (I7 → ℝ)) := by
  unfold μ7
  rw [volume_pi]
  exact pi_absolutelyContinuous (fun _ => gaussianReal_absolutelyContinuous 0 (by norm_num))

lemma gaussianReal_Iic_zero : gaussianReal 0 1 (Iic (0 : ℝ)) = (2 : ℝ≥0∞)⁻¹ := by
  let μ : Measure ℝ := gaussianReal 0 1
  letI : NoAtoms μ := noAtoms_gaussianReal (by norm_num)
  have hsymm : Measure.map (fun x : ℝ => -x) μ = μ := by
    simpa [μ] using gaussianReal_map_neg (μ := 0) (v := 1)
  have hleft_right : μ (Iic (0 : ℝ)) = μ (Ici (0 : ℝ)) := by
    calc
      μ (Iic (0 : ℝ)) = (Measure.map (fun x : ℝ => -x) μ) (Iic 0) := by rw [hsymm]
      _ = μ ((fun x : ℝ => -x) ⁻¹' Iic 0) := Measure.map_apply_of_aemeasurable
        measurable_neg.aemeasurable measurableSet_Iic
      _ = μ (Ici 0) := by congr 1; ext x; simp
  have hIci_Ioi : μ (Ici (0 : ℝ)) = μ (Ioi (0 : ℝ)) := by
    have hu : ({0} : Set ℝ) ∪ Ioi 0 = Ici 0 := by ext x; simp [le_iff_eq_or_lt]
    calc
      μ (Ici 0) = μ (({0} : Set ℝ) ∪ Ioi 0) := by rw [hu]
      _ = μ ({0} : Set ℝ) + μ (Ioi 0) := measure_union (by simp) measurableSet_Ioi
      _ = μ (Ioi 0) := by simp
  have hsum : μ (Iic (0 : ℝ)) + μ (Ioi (0 : ℝ)) = 1 := by
    calc
      μ (Iic 0) + μ (Ioi 0) = μ (Iic 0 ∪ Ioi 0) :=
        (measure_union (by simp) measurableSet_Ioi).symm
      _ = 1 := by simp
  have htwo : (2 : ℝ≥0∞) * μ (Iic (0 : ℝ)) = 1 := by
    rw [two_mul]
    simpa [hleft_right, hIci_Ioi] using hsum
  calc
    μ (Iic (0 : ℝ)) = (2 : ℝ≥0∞)⁻¹ * ((2 : ℝ≥0∞) * μ (Iic 0)) := by
      rw [ENNReal.inv_mul_cancel_left] <;> norm_num
    _ = (2 : ℝ≥0∞)⁻¹ := by rw [htwo, mul_one]

lemma gaussianReal_Ici_zero : gaussianReal 0 1 (Ici (0 : ℝ)) = (2 : ℝ≥0∞)⁻¹ := by
  have hsymm : gaussianReal 0 1 (Iic (0 : ℝ)) = gaussianReal 0 1 (Ici (0 : ℝ)) := by
    let μ : Measure ℝ := gaussianReal 0 1
    have hmap : Measure.map (fun x : ℝ => -x) μ = μ := by
      simpa [μ] using gaussianReal_map_neg (μ := 0) (v := 1)
    calc
      gaussianReal 0 1 (Iic 0) = (Measure.map (fun x : ℝ => -x) μ) (Iic 0) := by simpa [μ, hmap]
      _ = μ ((fun x : ℝ => -x) ⁻¹' Iic 0) := Measure.map_apply_of_aemeasurable
        measurable_neg.aemeasurable measurableSet_Iic
      _ = gaussianReal 0 1 (Ici 0) := by simp [μ]
  rw [← hsymm, gaussianReal_Iic_zero]

lemma zero_not_mem_interior_ratioCone (n : ℕ) :
    (0 : I7 → ℝ) ∉ interior (ratioCone n) := by
  classical
  intro h0
  rcases (Metric.isOpen_iff.1 isOpen_interior) 0 h0 with ⟨ε, hε, hball⟩
  let i0 : I7 := ⟨0, by simp⟩
  let i1 : I7 := ⟨1, by simp⟩
  let y : I7 → ℝ := Pi.single i0 (ε / 2)
  have hyball : y ∈ Metric.ball (0 : I7 → ℝ) ε := by
    rw [Metric.mem_ball, dist_zero_right, Pi.norm_single, Real.norm_eq_abs,
      abs_of_pos (half_pos hε)]
    linarith
  have hyint : y ∈ interior (ratioCone n) := hball hyball
  have hyD : y ∈ ratioCone n := interior_subset hyint
  have hy1 : y i1 = 0 := by
    simp [y, i0, i1]
  have hyzero : y = 0 := ratioCone_eq_zero_of_coord_eq_zero n hyD hy1
  have hy0 := congrFun hyzero i0
  simp [y, i0] at hy0
  linarith

def orthant (x : I7 → ℝ) : Set (I7 → ℝ) :=
  univ.pi fun i => if 0 < x i then Ici 0 else Iic 0

lemma μ7_orthant (x : I7 → ℝ) :
    μ7 (orthant x) = ((2 : ℝ≥0∞)⁻¹) ^ 7 := by
  classical
  simp [μ7, orthant, Measure.pi_pi, gaussianReal_Ici_zero, gaussianReal_Iic_zero]

end Bounty
