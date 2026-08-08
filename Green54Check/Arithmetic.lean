import Green54Check

open MeasureTheory ProbabilityTheory
open scoped Pointwise ENNReal NNReal Topology

namespace Green54Counterexample

-- Probe the exact representation of scientific literals in ENNReal/NNReal.

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
  apply (ENNReal.toReal_lt_toReal (by norm_num) (by norm_num)).mp
  rw [ENNReal.toReal_div]
  norm_num
  rw [show (1e-2 : ℝ≥0∞) = ((1e-2 : ℝ≥0) : ℝ≥0∞) from rfl]
  change ((1e-2 : ℝ≥0) : ℝ) = (1 : ℝ) / 100
  norm_num

end Green54Counterexample
