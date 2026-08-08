import Green54Check.Classification

open MeasureTheory ProbabilityTheory
open scoped Pointwise ENNReal NNReal Topology

namespace Green54Counterexample

local notation "γ" => Green54.gaussianMeasureInf

theorem green54_counterexample :
    ¬ (True ↔
      ∀ (K : Set Ω),
        IsCompact K →
          Balanced ℝ K →
            (0.99 : ℝ≥0∞) ≤ γ K →
              ∃ C : Set Ω,
                IsCompact C ∧
                  Convex ℝ C ∧
                    C ⊆ (10 : ℝ) • K ∧
                      (1e-2 : ℝ≥0∞) ≤ γ C) := by
  intro h
  have hP := h.mp trivial
  have hUtop : γ U ≠ ⊤ := by
    rw [measure_U]
    norm_num
  have h99 : (0.99 : ℝ≥0∞) < γ U := by
    rw [measure_U]
    rw [show (0.99 : ℝ≥0∞) = ((0.99 : ℝ≥0) : ℝ≥0∞) from rfl]
    exact_mod_cast (show (0.99 : ℝ≥0) < 1 by
      change ((0.99 : ℝ≥0) : ℝ) < 1
      norm_num)
  obtain ⟨B, hBU, hBcompact, hBmeasure⟩ :=
    measurableSet_U.exists_lt_isCompact_of_ne_top hUtop h99
  let K : Set Ω := scalarHull B
  have hKcompact : IsCompact K := by
    simpa [K] using isCompact_scalarHull hBcompact
  have hKbalanced : Balanced ℝ K := by
    simpa [K] using balanced_scalarHull B
  have hBK : B ⊆ K := by
    simpa [K] using subset_scalarHull B
  have hKmeasure : (0.99 : ℝ≥0∞) ≤ γ K :=
    (le_of_lt hBmeasure).trans (measure_mono hBK)
  obtain ⟨C, hCcompact, hCconvex, hCK, hCmeasure⟩ :=
    hP K hKcompact hKbalanced hKmeasure
  have hKshape : K ⊆ ({0} : Set Ω) ∪ U := by
    simpa [K] using scalarHull_subset_zero_union_U hBU
  have h10Kshape : (10 : ℝ) • K ⊆ ({0} : Set Ω) ∪ U :=
    smul_subset_zero_union_U hKshape 10
  have hCshape : C ⊆ ({0} : Set Ω) ∪ U := hCK.trans h10Kshape
  have hx : ∃ x ∈ C, x ≠ 0 := by
    by_contra hnone
    push_neg at hnone
    have hsub : C ⊆ ({0} : Set Ω) := by
      intro z hz
      simpa using hnone z hz
    have hm : γ C ≤ γ ({0} : Set Ω) := measure_mono hsub
    rw [measure_singleton_zero] at hm
    have hm0 : γ C = 0 := bot_unique hm
    have hpos : (0 : ℝ≥0∞) < (1e-2 : ℝ≥0∞) := by
      rw [show (1e-2 : ℝ≥0∞) = ((1e-2 : ℝ≥0) : ℝ≥0∞) from rfl]
      exact_mod_cast (show (0 : ℝ≥0) < (1e-2 : ℝ≥0) by
        change (0 : ℝ) < ((1e-2 : ℝ≥0) : ℝ)
        norm_num)
    have hpositiveC : (0 : ℝ≥0∞) < γ C := hpos.trans_le hCmeasure
    rw [hm0] at hpositiveC
    exact (lt_irrefl 0 hpositiveC)
  obtain ⟨x, hxC, hx0⟩ := hx
  have hxU : x ∈ U := by
    rcases hCshape hxC with hxzero | hxU
    · exact False.elim (hx0 (by simpa using hxzero))
    · exact hxU
  have hCcones : C ⊆ cone x ∪ cone (-x) :=
    convex_subset_two_cones hCconvex hCshape hxC hxU
  have hnegU : -x ∈ U := neg_mem_U hxU
  have hbound : γ C ≤ (1 / 128 : ℝ≥0∞) := by
    calc
      γ C ≤ γ (cone x ∪ cone (-x)) := measure_mono hCcones
      _ ≤ γ (cone x) + γ (cone (-x)) := measure_union_le (cone x) (cone (-x))
      _ ≤ (1 / 256 : ℝ≥0∞) + (1 / 256 : ℝ≥0∞) :=
        add_le_add (measure_cone_le hxU) (measure_cone_le hnegU)
      _ = (1 / 128 : ℝ≥0∞) := by
        apply (ENNReal.toReal_eq_toReal_iff' (by norm_num) (by norm_num)).mp
        rw [ENNReal.toReal_add (by norm_num) (by norm_num)]
        simp [ENNReal.toReal_div]
  have hbad : (1e-2 : ℝ≥0∞) ≤ (1 / 128 : ℝ≥0∞) :=
    hCmeasure.trans hbound
  have hlt : (1 / 128 : ℝ≥0∞) < (1e-2 : ℝ≥0∞) := by
    have hNN : (1 / 128 : ℝ≥0) < (1e-2 : ℝ≥0) := by
      change ((1 / 128 : ℝ≥0) : ℝ) < ((1e-2 : ℝ≥0) : ℝ)
      norm_num
    calc
      (1 / 128 : ℝ≥0∞) = ((1 / 128 : ℝ≥0) : ℝ≥0∞) := by
        symm
        exact ENNReal.coe_div (by norm_num : (128 : ℝ≥0) ≠ 0)
      _ < ((1e-2 : ℝ≥0) : ℝ≥0∞) := by exact_mod_cast hNN
      _ = (1e-2 : ℝ≥0∞) := rfl
  exact (not_le_of_gt hlt) hbad

#print axioms green54_counterexample

end Green54Counterexample
