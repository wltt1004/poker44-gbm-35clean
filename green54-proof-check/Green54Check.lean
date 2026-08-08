import Mathlib
import Mathlib.MeasureTheory.Measure.Lebesgue.EqHaar
import Mathlib.Analysis.Convex.Measure

open MeasureTheory ProbabilityTheory Set
open scoped Pointwise ENNReal

#check MeasurableSet.exists_lt_isCompact_of_ne_top
#check gaussianReal_absolutelyContinuous
#check noAtoms_gaussianReal
#check gaussianReal_map_neg
#check Measure.pi'_eq_pi
#check Measure.tprod_cons
#check Measure.AbsolutelyContinuous.prod
#check Measure.AbsolutelyContinuous.map
#check Measure.infinitePi_cylinder
#check Measure.infinitePi_map_restrict
#check Measure.pi_pi
#check tendsto_measure_iUnion_atTop
#check Convex.interior_nonempty_iff_affineSpan_eq_top
#check Measure.addHaar_affineSubspace
#check Measure.addHaar_submodule
#check IsCompact.smul_set
#check isCompact_Icc
#check isCompact_closedBall
#check isCompact_univ_pi
#check isClosed_iInter
#check isClosed_le
#check continuous_apply
#check Continuous.abs
#check Balanced
#check Balanced.smul_mem
#check balancedHull.balanced
#check subset_balancedHull
#check Balanced.balancedHull_subset_of_subset
#check mem_balancedHull_iff
#check Set.mem_smul_set
#check measure_union
#check Convex.lineMap_mem
#check Convex.segment_subset
#check Finite.exists_le
#check Finite.exists_ge
#check exists_nat_ge
#check exists_nat_gt
#check finrank_span_singleton
#check Module.finrank_pi
#check Module.finrank_fin_fun
#check affineSpan_le
#check Submodule.span_le
#check Measure.map_apply
#check Measure.map_apply_of_aemeasurable
#check Measure.restrict_apply
#check MeasurableSet.cylinder
#check continuous_finset_restrict
#check Finset.continuous_restrict
#check Finset.restrict
#check Convex.linear_image
#check Convex.isometry_image
#check Measure.AbsolutelyContinuous.measure_eq_zero
#check Measure.AbsolutelyContinuous.mk
#check MeasurableSet.Iic
#check MeasurableSet.Ici
#check MeasurableSet.singleton
#check measure_singleton
#check MeasureTheory.Measure.measure_singleton
#check Measure.map_mono
#check Measure.mono
#check Filter.Tendsto.eventually
#check Metric.isOpen_iff

namespace Bounty

example : True := by trivial

end Bounty
