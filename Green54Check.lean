import FormalConjectures.GreensOpenProblems.«54»

/-! Exact-environment development for the Green 54 counterexample task. -/

open MeasureTheory ProbabilityTheory
open scoped Pointwise ENNReal NNReal Topology

namespace Green54Counterexample

abbrev Ω := ℕ → ℝ
local notation "γ" => Green54.gaussianMeasureInf
local notation "g" => gaussianReal 0 1

def U : Set Ω := Set.univ.pi (fun _ : ℕ => ({0} : Set ℝ)ᶜ)

lemma mem_U {x : Ω} : x ∈ U ↔ ∀ n, x n ≠ 0 := by
  simp [U]

lemma measurableSet_U : MeasurableSet U := by
  apply MeasurableSet.pi Set.countable_univ
  intro i hi
  measurability

lemma measure_U : γ U = 1 := by
  letI : NoAtoms g := noAtoms_gaussianReal (μ := 0) (v := 1) (by norm_num)
  have hfactor : g (({0} : Set ℝ)ᶜ) = 1 := by
    rw [measure_compl (measurableSet_singleton 0) (measure_ne_top g {0}), measure_univ]
    simp
  rw [Green54.gaussianMeasureInf, U,
    Measure.infinitePi_pi_univ (fun _ : ℕ => g) (by intro i; measurability)]
  simp [hfactor]

lemma gaussian_half_closed :
    g (Set.Ici (0 : ℝ)) = (1 / 2 : ℝ≥0∞) ∧
      g (Set.Iic (0 : ℝ)) = (1 / 2 : ℝ≥0∞) := by
  letI : NoAtoms g := noAtoms_gaussianReal (μ := 0) (v := 1) (by norm_num)
  have hmap : Measure.map (fun x : ℝ => -x) g = g := by
    simpa using (gaussianReal_map_neg (μ := (0 : ℝ)) (v := (1 : ℝ≥0)))
  have hsym : g (Set.Ici (0 : ℝ)) = g (Set.Iic (0 : ℝ)) := by
    calc
      g (Set.Ici (0 : ℝ)) = (Measure.map (fun x : ℝ => -x) g) (Set.Ici (0 : ℝ)) := by
        rw [hmap]
      _ = g ((fun x : ℝ => -x) ⁻¹' Set.Ici (0 : ℝ)) := by
        rw [Measure.map_apply (by fun_prop) measurableSet_Ici]
      _ = g (Set.Iic (0 : ℝ)) := by
        congr 1
        ext x
        simp
  have hIic : g (Set.Iic (0 : ℝ)) = g (Set.Iio (0 : ℝ)) := by
    have hs : Set.Iic (0 : ℝ) = Set.Iio (0 : ℝ) ∪ {0} := by
      ext x
      simp only [Set.mem_Iic, Set.mem_union, Set.mem_Iio, Set.mem_singleton_iff]
      exact le_iff_lt_or_eq
    have hdis : Disjoint (Set.Iio (0 : ℝ)) ({0} : Set ℝ) := by
      refine Set.disjoint_left.2 ?_
      intro x hx h0
      change x < 0 at hx
      change x = 0 at h0
      subst x
      exact (lt_irrefl 0 hx)
    rw [hs, measure_union hdis (measurableSet_singleton 0)]
    simp
  have hsum : g (Set.Iio (0 : ℝ)) + g (Set.Ici (0 : ℝ)) = 1 := by
    have hdis : Disjoint (Set.Iio (0 : ℝ)) (Set.Ici (0 : ℝ)) := by
      refine Set.disjoint_left.2 ?_
      intro x hx₁ hx₂
      change x < 0 at hx₁
      change 0 ≤ x at hx₂
      exact (not_lt_of_ge hx₂) hx₁
    rw [← measure_union hdis measurableSet_Ici]
    have hs : Set.Iio (0 : ℝ) ∪ Set.Ici (0 : ℝ) = Set.univ := by
      ext x
      simp
    rw [hs, measure_univ]
  have hsame : g (Set.Ici (0 : ℝ)) = g (Set.Iio (0 : ℝ)) := hsym.trans hIic
  have hdouble : g (Set.Ici (0 : ℝ)) + g (Set.Ici (0 : ℝ)) = 1 := by
    calc
      g (Set.Ici (0 : ℝ)) + g (Set.Ici (0 : ℝ))
          = g (Set.Iio (0 : ℝ)) + g (Set.Ici (0 : ℝ)) := by rw [hsame]
      _ = 1 := hsum
  have ha_top : g (Set.Ici (0 : ℝ)) ≠ ⊤ := measure_ne_top g _
  have hreal : (g (Set.Ici (0 : ℝ))).toReal + (g (Set.Ici (0 : ℝ))).toReal = 1 := by
    rw [← ENNReal.toReal_add ha_top ha_top, hdouble]
    norm_num
  have hhalf_real : (g (Set.Ici (0 : ℝ))).toReal = (1 / 2 : ℝ) := by
    linarith
  have hhalf : g (Set.Ici (0 : ℝ)) = (1 / 2 : ℝ≥0∞) := by
    apply (ENNReal.toReal_eq_toReal_iff' ha_top (by norm_num)).mp
    simpa using hhalf_real
  exact ⟨hhalf, hsym.symm.trans hhalf⟩

def halfLine (a : ℝ) : Set ℝ :=
  if 0 ≤ a then Set.Ici 0 else Set.Iic 0

lemma measurableSet_halfLine (a : ℝ) : MeasurableSet (halfLine a) := by
  unfold halfLine
  split_ifs <;> measurability

lemma measure_halfLine (a : ℝ) : g (halfLine a) = (1 / 2 : ℝ≥0∞) := by
  rcases gaussian_half_closed with ⟨hpos, hneg⟩
  by_cases h : 0 ≤ a
  · simpa [halfLine, h] using hpos
  · simpa [halfLine, h] using hneg

def orthant8 (x : Ω) : Set Ω :=
  Set.pi (Finset.range 8) (fun i => halfLine (x i))

def cone (x : Ω) : Set Ω :=
  {y | ∀ i, 0 ≤ x i * y i}

lemma measure_orthant8 (x : Ω) :
    γ (orthant8 x) = (1 / 256 : ℝ≥0∞) := by
  rw [Green54.gaussianMeasureInf, orthant8,
    Measure.infinitePi_pi (fun _ : ℕ => g)
      (fun i hi => measurableSet_halfLine (x i))]
  norm_num [measure_halfLine, inv_pow]

lemma cone_subset_orthant8 {x : Ω} (hx : x ∈ U) :
    cone x ⊆ orthant8 x := by
  intro y hy
  change ∀ i, 0 ≤ x i * y i at hy
  change ∀ i ∈ Finset.range 8, y i ∈ halfLine (x i)
  intro i hi
  by_cases hxi : 0 ≤ x i
  · have hxpos : 0 < x i := lt_of_le_of_ne hxi (Ne.symm ((mem_U.mp hx) i))
    simpa [halfLine, hxi] using nonneg_of_mul_nonneg_right (hy i) hxpos
  · have hxneg : x i < 0 := lt_of_not_ge hxi
    simpa [halfLine, hxi] using nonpos_of_mul_nonneg_right (hy i) hxneg

lemma measure_cone_le {x : Ω} (hx : x ∈ U) :
    γ (cone x) ≤ (1 / 256 : ℝ≥0∞) := by
  calc
    γ (cone x) ≤ γ (orthant8 x) := measure_mono (cone_subset_orthant8 hx)
    _ = (1 / 256 : ℝ≥0∞) := measure_orthant8 x

lemma neg_mem_U {x : Ω} (hx : x ∈ U) : -x ∈ U := by
  rw [mem_U] at hx ⊢
  intro n
  simpa using neg_ne_zero.mpr (hx n)

lemma measure_singleton_zero : γ ({0} : Set Ω) = 0 := by
  letI : NoAtoms g := noAtoms_gaussianReal (μ := 0) (v := 1) (by norm_num)
  let Z : Set Ω := Set.pi ({0} : Finset ℕ) (fun _ => ({0} : Set ℝ))
  have hZ : γ Z = 0 := by
    rw [Green54.gaussianMeasureInf, Z,
      Measure.infinitePi_pi (fun _ : ℕ => g)
        (fun i hi => measurableSet_singleton 0)]
    simp
  apply measure_mono_null ?_ hZ
  intro x hx
  have hx0 : x = 0 := by simpa using hx
  subst x
  simp [Z]

def scalarHull (B : Set Ω) : Set Ω :=
  Metric.closedBall (0 : ℝ) 1 • B

lemma subset_scalarHull (B : Set Ω) : B ⊆ scalarHull B := by
  intro x hx
  have h1 : (1 : ℝ) ∈ Metric.closedBall (0 : ℝ) 1 := by simp
  simpa [scalarHull] using Set.smul_mem_smul h1 hx

lemma isCompact_scalarHull {B : Set Ω} (hB : IsCompact B) :
    IsCompact (scalarHull B) := by
  exact IsCompact.smul_set isCompact_closedBall hB

lemma balanced_scalarHull (B : Set Ω) : Balanced ℝ (scalarHull B) := by
  intro a ha
  rintro z ⟨y, hy, rfl⟩
  rcases hy with ⟨r, hr, x, hx, rfl⟩
  have hrnorm : ‖r‖ ≤ 1 := by
    simpa [Metric.mem_closedBall, dist_eq_norm] using hr
  have harnorm : ‖a * r‖ ≤ 1 :=
    (norm_mul_le a r).trans (mul_le_one₀ ha (norm_nonneg r) hrnorm)
  have har : a * r ∈ Metric.closedBall (0 : ℝ) 1 := by
    simpa [Metric.mem_closedBall, dist_eq_norm] using harnorm
  refine ⟨a * r, har, x, hx, ?_⟩
  simp [smul_smul]

lemma scalarHull_subset_zero_union_U {B : Set Ω} (hB : B ⊆ U) :
    scalarHull B ⊆ ({0} : Set Ω) ∪ U := by
  rintro z ⟨r, hr, x, hx, rfl⟩
  by_cases hr0 : r = 0
  · left
    simp [hr0]
  · right
    rw [mem_U]
    intro n
    change r * x n ≠ 0
    exact mul_ne_zero hr0 ((mem_U.mp (hB hx)) n)

lemma smul_subset_zero_union_U {S : Set Ω}
    (hS : S ⊆ ({0} : Set Ω) ∪ U) (a : ℝ) :
    a • S ⊆ ({0} : Set Ω) ∪ U := by
  rintro z ⟨x, hx, rfl⟩
  rcases hS hx with hx0 | hxU
  · left
    have hzero : x = 0 := by simpa using hx0
    simp [hzero]
  · by_cases ha : a = 0
    · left
      simp [ha]
    · right
      rw [mem_U]
      intro n
      change a * x n ≠ 0
      exact mul_ne_zero ha ((mem_U.mp hxU) n)

end Green54Counterexample
