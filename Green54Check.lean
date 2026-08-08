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
  rw [Green54.gaussianMeasureInf, U,
    Measure.infinitePi_pi_univ (fun _ : ℕ => g) (by intro i; measurability)]
  simp

#check measure_union
#check ENNReal.toReal_add
#check measure_ne_top
#check isCompact_closedBall
#check Metric.isCompact_closedBall
#check Convex

lemma gaussian_half_closed :
    g (Set.Ici (0 : ℝ)) = (1 / 2 : ℝ≥0∞) ∧
      g (Set.Iic (0 : ℝ)) = (1 / 2 : ℝ≥0∞) := by
  letI : NoAtoms g := noAtoms_gaussianReal (μ := 0) (v := 1) (by norm_num)
  have hmap : g.map (fun x : ℝ => -x) = g := by
    simpa using (gaussianReal_map_neg (μ := (0 : ℝ)) (v := (1 : ℝ≥0)))
  have hsym : g (Set.Ici (0 : ℝ)) = g (Set.Iic (0 : ℝ)) := by
    calc
      g (Set.Ici (0 : ℝ)) = (g.map (fun x : ℝ => -x)) (Set.Ici (0 : ℝ)) := by rw [hmap]
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
    rw [hs, measure_union]
    · simp
    · exact Set.disjoint_left.2 (by
        intro x hx h0
        simpa [h0] using hx)
    · measurability
  have hsum : g (Set.Iio (0 : ℝ)) + g (Set.Ici (0 : ℝ)) = 1 := by
    rw [← measure_union]
    · have hs : Set.Iio (0 : ℝ) ∪ Set.Ici (0 : ℝ) = Set.univ := by
        ext x
        simp
      rw [hs, measure_univ]
    · exact Set.disjoint_left.2 (by
        intro x hx₁ hx₂
        exact (not_lt_of_ge hx₂) hx₁)
    · measurability
  have hdouble : g (Set.Ici (0 : ℝ)) + g (Set.Ici (0 : ℝ)) = 1 := by
    calc
      g (Set.Ici (0 : ℝ)) + g (Set.Ici (0 : ℝ))
          = g (Set.Iio (0 : ℝ)) + g (Set.Ici (0 : ℝ)) := by rw [← hIic, ← hsym]
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
  exact ⟨hhalf, hsym ▸ hhalf⟩

end Green54Counterexample
