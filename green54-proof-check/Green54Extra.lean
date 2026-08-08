
namespace Bounty

noncomputable def γ∞ : Measure (ℕ → ℝ) :=
  Measure.infinitePi (fun _ : ℕ => gaussianReal 0 1)

noncomputable instance γ∞_isProbabilityMeasure : IsProbabilityMeasure γ∞ := by
  unfold γ∞
  infer_instance

lemma γ∞_cylinder_ratioCone (n : ℕ) :
    γ∞ (cylinder (Finset.range 7) (ratioCone n)) = μ7 (ratioCone n) := by
  simpa [γ∞, μ7] using
    (Measure.infinitePi_cylinder (fun _ : ℕ => gaussianReal 0 1)
      (ratioCone_measurable n))

lemma cylinder_ratioCone_smul_mem (n : ℕ) {x : ℕ → ℝ}
    (hx : x ∈ cylinder (Finset.range 7) (ratioCone n)) (a : ℝ) :
    a • x ∈ cylinder (Finset.range 7) (ratioCone n) := by
  rw [mem_cylinder] at hx ⊢
  simpa [Finset.restrict_def, Pi.smul_apply] using ratioCone_smul_mem n hx a

def restrictLinear (s : Finset ℕ) : (ℕ → ℝ) →ₗ[ℝ] (↥s → ℝ) where
  toFun := s.restrict
  map_add' x y := by
    ext i
    rfl
  map_smul' c x := by
    ext i
    rfl

lemma small_orthant_bound : ((2 : ℝ≥0∞)⁻¹) ^ 7 < (0.01 : ℝ≥0∞) := by
  rw [NNRatCast.toOfScientific_def]
  norm_num

theorem no_scalar_green54 :
    ¬ (∀ K : Set (ℕ → ℝ), IsCompact K → Balanced ℝ K →
      (0.99 : ℝ≥0∞) ≤ γ∞ K →
      ∃ C : Set (ℕ → ℝ), IsCompact C ∧ Convex ℝ C ∧
        C ⊆ (10 : ℝ) • K ∧ (0.01 : ℝ≥0∞) ≤ γ∞ C) := by
  intro htarget
  obtain ⟨n, hn⟩ := exists_ratioCone_large
  let A : Set (ℕ → ℝ) := cylinder (Finset.range 7) (ratioCone n)
  have hAmeas : MeasurableSet A := by
    exact (ratioCone_measurable n).cylinder (Finset.range 7)
  have hAlarge : (0.99 : ℝ≥0∞) < γ∞ A := by
    simpa [A, γ∞_cylinder_ratioCone] using hn
  have hAne : γ∞ A ≠ ⊤ := measure_ne_top γ∞ A
  obtain ⟨B, hBA, hBcompact, hBlarge⟩ :=
    hAmeas.exists_lt_isCompact_of_ne_top hAne hAlarge
  let K : Set (ℕ → ℝ) := Metric.closedBall (0 : ℝ) 1 • B
  have hKcompact : IsCompact K := by
    exact (isCompact_closedBall (0 : ℝ) 1).smul_set hBcompact
  have hBsubsetK : B ⊆ K := by
    intro b hb
    refine ⟨1, ?_, b, hb, ?_⟩
    · simp [Metric.mem_closedBall]
    · simp
  have hKsubsetA : K ⊆ A := by
    intro z hz
    rcases hz with ⟨a, ha, b, hb, rfl⟩
    exact cylinder_ratioCone_smul_mem n (hBA hb) a
  have hKbalanced : Balanced ℝ K := by
    rw [balanced_iff_smul_mem]
    intro c hc z hz
    rcases hz with ⟨a, ha, b, hb, rfl⟩
    refine ⟨c * a, ?_, b, hb, ?_⟩
    · have ha' : ‖a‖ ≤ 1 := by
        simpa [Metric.mem_closedBall, dist_zero_right] using ha
      simp only [Metric.mem_closedBall, dist_zero_right, norm_mul]
      calc
        ‖c‖ * ‖a‖ ≤ 1 * 1 := mul_le_mul hc ha' (norm_nonneg a) (by norm_num)
        _ = 1 := by norm_num
    · simp [mul_smul]
  have hKlarge : (0.99 : ℝ≥0∞) ≤ γ∞ K :=
    (le_of_lt hBlarge).trans (measure_mono hBsubsetK)
  obtain ⟨C, hCcompact, hCconvex, hCsub10K, hClarge⟩ :=
    htarget K hKcompact hKbalanced hKlarge
  have h10KsubsetA : (10 : ℝ) • K ⊆ A := by
    intro z hz
    rcases Set.mem_smul_set.mp hz with ⟨k, hk, rfl⟩
    exact cylinder_ratioCone_smul_mem n (hKsubsetA hk) 10
  have hCsubsetA : C ⊆ A := hCsub10K.trans h10KsubsetA
  let P : Set (I7 → ℝ) := restrictLinear (Finset.range 7) '' C
  have hPcompact : IsCompact P := by
    simpa [P, restrictLinear] using
      hCcompact.image (Finset.continuous_restrict (Finset.range 7))
  have hPconvex : Convex ℝ P := by
    simpa [P] using hCconvex.linear_image (restrictLinear (Finset.range 7))
  have hPsubset : P ⊆ ratioCone n := by
    rintro p ⟨c, hc, rfl⟩
    have hcA := hCsubsetA hc
    rw [mem_cylinder] at hcA
    simpa [restrictLinear] using hcA
  have hCsubsetCylinder : C ⊆ cylinder (Finset.range 7) P := by
    intro c hc
    rw [mem_cylinder]
    change restrictLinear (Finset.range 7) c ∈ P
    exact ⟨c, hc, rfl⟩
  have hPcylinder :
      γ∞ (cylinder (Finset.range 7) P) = μ7 P := by
    simpa [γ∞, μ7] using
      (Measure.infinitePi_cylinder (fun _ : ℕ => gaussianReal 0 1)
        hPcompact.measurableSet)
  have hCupper : γ∞ C ≤ ((2 : ℝ≥0∞)⁻¹) ^ 7 := by
    calc
      γ∞ C ≤ γ∞ (cylinder (Finset.range 7) P) := measure_mono hCsubsetCylinder
      _ = μ7 P := hPcylinder
      _ ≤ ((2 : ℝ≥0∞)⁻¹) ^ 7 :=
        convex_subset_ratioCone_measure_le n hPconvex hPsubset
  exact (not_le_of_gt small_orthant_bound) (hClarge.trans hCupper)

end Bounty
