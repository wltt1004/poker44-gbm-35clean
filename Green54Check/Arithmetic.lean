import Green54Check

/-! Probe the exact representation of scientific literals in `ENNReal` and `NNReal`. -/

open MeasureTheory ProbabilityTheory
open scoped Pointwise ENNReal NNReal Topology

namespace Green54Counterexample

example : (0.99 : ℝ≥0∞) = ((0.99 : ℝ≥0) : ℝ≥0∞) := by rfl
example : (1e-2 : ℝ≥0∞) = ((1e-2 : ℝ≥0) : ℝ≥0∞) := by rfl

example : ((0.99 : ℝ≥0) : ℝ) = (0.99 : ℝ) := by rfl
example : ((1e-2 : ℝ≥0) : ℝ) = (1e-2 : ℝ) := by rfl

example : (0.99 : ℝ≥0) < 1 := by
  change ((0.99 : ℝ≥0) : ℝ) < 1
  norm_num

example : (0 : ℝ≥0) < (1e-2 : ℝ≥0) := by
  change (0 : ℝ) < ((1e-2 : ℝ≥0) : ℝ)
  norm_num

example : (0.99 : ℝ≥0∞) < 1 := by
  rw [show (0.99 : ℝ≥0∞) = ((0.99 : ℝ≥0) : ℝ≥0∞) from rfl]
  exact_mod_cast (show (0.99 : ℝ≥0) < 1 by
    change ((0.99 : ℝ≥0) : ℝ) < 1
    norm_num)

example : (0 : ℝ≥0∞) < (1e-2 : ℝ≥0∞) := by
  rw [show (1e-2 : ℝ≥0∞) = ((1e-2 : ℝ≥0) : ℝ≥0∞) from rfl]
  exact_mod_cast (show (0 : ℝ≥0) < (1e-2 : ℝ≥0) by
    change (0 : ℝ) < ((1e-2 : ℝ≥0) : ℝ)
    norm_num)

example : (0.99 : ℝ≥0∞).toReal = (99 : ℝ) / 100 := by
  rw [show (0.99 : ℝ≥0∞) = ((0.99 : ℝ≥0) : ℝ≥0∞) from rfl]
  change ((0.99 : ℝ≥0) : ℝ) = (99 : ℝ) / 100
  norm_num

example : (1e-2 : ℝ≥0∞).toReal = (1 : ℝ) / 100 := by
  rw [show (1e-2 : ℝ≥0∞) = ((1e-2 : ℝ≥0) : ℝ≥0∞) from rfl]
  change ((1e-2 : ℝ≥0) : ℝ) = (1 : ℝ) / 100
  norm_num

example : (1 / 128 : ℝ≥0∞) < (1e-2 : ℝ≥0∞) := by
  have hNN : (1 / 128 : ℝ≥0) < (1e-2 : ℝ≥0) := by
    change ((1 / 128 : ℝ≥0) : ℝ) < ((1e-2 : ℝ≥0) : ℝ)
    norm_num
  calc
    (1 / 128 : ℝ≥0∞) = ((1 / 128 : ℝ≥0) : ℝ≥0∞) := by
      symm
      exact ENNReal.coe_div (by norm_num : (128 : ℝ≥0) ≠ 0)
    _ < ((1e-2 : ℝ≥0) : ℝ≥0∞) := by exact_mod_cast hNN
    _ = (1e-2 : ℝ≥0∞) := rfl

end Green54Counterexample
