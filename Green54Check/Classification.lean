import Green54Check

/-! A convex set contained in `{0} ∪ U` lies in two opposite sign cones. -/

open MeasureTheory ProbabilityTheory
open scoped Pointwise ENNReal NNReal Topology

namespace Green54Counterexample

lemma convex_subset_two_cones {C : Set Ω} (hconv : Convex ℝ C)
    (hshape : C ⊆ ({0} : Set Ω) ∪ U)
    {x : Ω} (hxC : x ∈ C) (hxU : x ∈ U) :
    C ⊆ cone x ∪ cone (-x) := by
  intro y hyC
  rcases hshape hyC with hy0 | hyU
  · left
    change ∀ i, 0 ≤ x i * y i
    have hyzero : y = 0 := by simpa using hy0
    subst y
    simp
  · by_cases hall : ∀ i, 0 ≤ x i * y i
    · exact Or.inl hall
    · have hex : ∃ n, x n * y n < 0 := by
        push_neg at hall
        exact hall
      obtain ⟨n, hn⟩ := hex
      have hden : x n - y n ≠ 0 := by
        intro hzero
        have heq : x n = y n := sub_eq_zero.mp hzero
        rw [heq] at hn
        nlinarith [sq_nonneg (y n)]
      let r : ℝ := -y n / (x n - y n)
      let s : ℝ := x n / (x n - y n)
      have hrs : 0 < r ∧ 0 < s := by
        dsimp [r, s]
        by_cases hxp : 0 < x n
        · have hyn : y n < 0 := by
            by_contra hnot
            have hy_nonneg : 0 ≤ y n := le_of_not_gt hnot
            exact (not_lt_of_ge (mul_nonneg hxp.le hy_nonneg)) hn
          have hdenpos : 0 < x n - y n := by linarith
          exact ⟨div_pos (by linarith) hdenpos, div_pos hxp hdenpos⟩
        · have hx_nonpos : x n ≤ 0 := le_of_not_gt hxp
          have hxneg : x n < 0 :=
            lt_of_le_of_ne hx_nonpos ((mem_U.mp hxU) n)
          have hyp : 0 < y n := by
            by_contra hnot
            have hy_nonpos : y n ≤ 0 := le_of_not_gt hnot
            exact (not_lt_of_ge
              (mul_nonneg_of_nonpos_of_nonpos hxneg.le hy_nonpos)) hn
          have hdenneg : x n - y n < 0 := by linarith
          exact ⟨div_pos_of_neg_of_neg (by linarith) hdenneg,
            div_pos_of_neg_of_neg hxneg hdenneg⟩
      have hrsum : r + s = 1 := by
        dsimp [r, s]
        field_simp [hden]
        ring
      have hcancel : r * x n + s * y n = 0 := by
        dsimp [r, s]
        field_simp [hden]
        ring
      have hwC : r • x + s • y ∈ C :=
        hconv hxC hyC hrs.1.le hrs.2.le hrsum
      have hwn : (r • x + s • y) n = 0 := by
        change r * x n + s * y n = 0
        exact hcancel
      have hw0 : r • x + s • y = 0 := by
        rcases hshape hwC with hwzero | hwU
        · simpa using hwzero
        · exact False.elim ((mem_U.mp hwU n) hwn)
      right
      change ∀ i, 0 ≤ (-x) i * y i
      intro i
      have hcoord : r * x i + s * y i = 0 := by
        have hfun := congrFun hw0 i
        simpa using hfun
      have hmul : s * (x i * y i) = -r * (x i) ^ 2 := by
        have hsy : s * y i = -r * x i := by linarith
        calc
          s * (x i * y i) = x i * (s * y i) := by ring
          _ = x i * (-r * x i) := by rw [hsy]
          _ = -r * (x i) ^ 2 := by ring
      have hxy : x i * y i ≤ 0 := by
        nlinarith [sq_nonneg (x i)]
      change 0 ≤ (-x i) * y i
      nlinarith

end Green54Counterexample
