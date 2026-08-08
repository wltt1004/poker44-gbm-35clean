import FormalConjectures.GreensOpenProblems.«54»

/-! Exact-environment development for the Green 54 counterexample task. -/

open MeasureTheory ProbabilityTheory
open scoped Pointwise ENNReal Topology

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
    rw [measure_compl (by measurability), measure_univ]
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
        rw [Measure.map_apply (by fun_prop) (by measurability)]
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
    rw [hs, measure_union hdis (by measurability)]
    simp
  have hsum : g (Set.Iio (0 : ℝ)) + g (Set.Ici (0 : ℝ)) = 1 := by
    have hdis : Disjoint (Set.Iio (0 : ℝ)) (Set.Ici (0 : ℝ)) := by
      refine Set.disjoint_left.2 ?_
      intro x hx₁ hx₂
      change x < 0 at hx₁
      change 0 ≤ x at hx₂
      exact (not_lt_of_ge hx₂) hx₁
    rw [← measure_union hdis (by measurability)]
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

end Green54Counterexample
