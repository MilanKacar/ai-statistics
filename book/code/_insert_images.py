"""
Internal helper (not a chapter script): inserts a '## Figures' section with
the generated images (and captions) into each chapter markdown file, right
before the '## Run it yourself' heading. Idempotent: if a chapter already has
a '## Figures' section, it's skipped.
"""
import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOK_DIR = os.path.dirname(CODE_DIR)
CHAPTERS_DIR = os.path.join(BOOK_DIR, "chapters")

# chapter_file -> list of (image_filename, caption)
FIGURES = {
    "01_data_populations_samples.md": [
        ("ch01_population_vs_sample_fig1.png",
         "Sample means from the same population, drawn at n=5, 30, 200. Notice the spread shrinks as n grows -- the seed of the Central Limit Theorem (Chapter 8)."),
    ],
    "02_descriptive_statistics.md": [
        ("ch02_descriptive_stats_fig1.png",
         "A right-skewed income-like dataset: the histogram with mean and median marked, and the corresponding box plot. Notice how far the mean sits from the median."),
        ("ch02_descriptive_stats_fig2.png",
         "Anscombe's Quartet: four datasets with identical mean, variance, correlation, and regression line -- yet visibly different shapes."),
    ],
    "03_probability_foundations.md": [
        ("ch03_probability_foundations_fig1.png",
         "Law of Large Numbers: the empirical frequency of rolling a 4 converges to the true probability 1/6 as the number of rolls grows (x-axis is log-scaled)."),
    ],
    "04_bayes_theorem.md": [
        ("ch04_bayes_theorem_fig1.png",
         "How the posterior P(disease | positive test) changes as the prior (disease prevalence) sweeps from near 0 to 1, for a 99%-accurate test. Our worked example (prior=0.01) is marked."),
    ],
    "05_random_variables_discrete.md": [
        ("ch05_discrete_distributions_fig1.png",
         "Top row: theoretical PMFs for Binomial(n=10, p=0.3) and Poisson(lambda=3). Bottom row: 50,000 simulated draws from each, overlaid on the theoretical PMF to confirm they match."),
    ],
    "06_continuous_distributions.md": [
        ("ch06_continuous_distributions_fig1.png",
         "The Uniform, Exponential, and Normal probability density functions side by side."),
        ("ch06_continuous_distributions_fig2.png",
         "The 68-95-99.7 rule for the Standard Normal distribution: the shaded regions show the area (probability) within 1, 2, and 3 standard deviations of the mean."),
    ],
    "07_expectation_variance_covariance.md": [
        ("ch07_expectation_variance_covariance_fig1.png",
         "Four (X, Y) relationships and their computed correlation rho: strong positive, strong negative, none, and a perfect parabola -- notice the parabola's near-zero rho despite Y being completely determined by X."),
    ],
    "08_central_limit_theorem.md": [
        ("ch08_central_limit_theorem_fig1.png",
         "Starting from a heavily right-skewed exponential population, the distribution of the sample mean becomes an increasingly clean bell curve as sample size n grows from 1 to 100."),
    ],
    "09_sampling_and_estimation.md": [
        ("ch09_sampling_and_estimation_fig1.png",
         "Sampling distributions of three estimators (sample mean, sample median, first data point) across three sample sizes. The red dashed line marks the true population mean -- compare how tightly each estimator clusters around it."),
    ],
    "10_maximum_likelihood_estimation.md": [
        ("ch10_maximum_likelihood_estimation_fig1.png",
         "The log-likelihood of Normal data as a function of candidate mu values. The peak sits exactly at the sample mean -- proof that the MLE for a Normal mean IS the sample mean."),
    ],
    "11_confidence_intervals.md": [
        ("ch11_confidence_intervals_fig1.png",
         "100 independently constructed 95% confidence intervals from repeated samples of a known population. About 95 of them (blue) capture the true mean; a handful (red) miss it -- this is what '95% confidence' actually means."),
    ],
    "12_hypothesis_testing.md": [
        ("ch12_hypothesis_testing_fig1.png",
         "p-values from 20 completely unrelated hypothesis tests run on pure noise (the null hypothesis is true for every single one). Notice p-values are uniformly distributed -- some will always look 'significant' by chance alone."),
    ],
    "13_ttests_chisquare_anova.md": [
        ("ch13_ttests_chisquare_anova_fig1.png",
         "The t-distribution (heavier tails at low degrees of freedom, converging to Normal), the chi-square distribution (for varying k), and the F-distribution (ratio of two variance estimates), all plotted directly."),
    ],
    "14_simple_linear_regression.md": [
        ("ch14_simple_linear_regression_fig1.png",
         "The hours-studied vs exam-score example: data points with the least-squares best-fit line and its equation."),
        ("ch14_simple_linear_regression_fig2.png",
         "Anscombe's Quartet refit with linear regression. Top row: fitted lines (identical R^2 for all four). Bottom row: residual plots, which instantly expose the curved relationship, the outlier, and the degenerate x-values that R^2 alone hides."),
    ],
    "15_multiple_regression.md": [
        ("ch15_multiple_regression_fig1.png",
         "Raw R^2 keeps climbing as more and more USELESS random-noise predictors are added to the model, while adjusted R^2 stays flat or declines -- overfitting made mechanically visible."),
    ],
    "16_bias_variance_regularization.md": [
        ("ch16_bias_variance_regularization_fig1.png",
         "Training error (keeps improving) vs cross-validated error (U-shaped) as polynomial degree increases. The green line marks the complexity level cross-validation selects as best."),
        ("ch16_bias_variance_regularization_fig2.png",
         "A wildly overfit degree-14 polynomial (red) tamed by Ridge regularization at increasing lambda values, pulling the curve back toward the true underlying function (black dashed)."),
    ],
    "17_bayesian_statistics.md": [
        ("ch17_bayesian_statistics_fig1.png",
         "The posterior distribution for a coin's true bias p, sharpening and shifting toward the true value (black dashed line) as more coin flips accumulate -- the prior's influence visibly fades with more data."),
    ],
    "18_resampling_methods.md": [
        ("ch18_resampling_methods_fig1.png",
         "The bootstrap distribution of the sample median, built from 10,000 resamples with replacement, with the 95% percentile-based confidence interval marked."),
        ("ch18_resampling_methods_fig2.png",
         "A permutation test's null distribution (differences in conversion rate under randomly shuffled group labels), with the real observed difference marked in red."),
    ],
    "19_linear_algebra_essentials.md": [
        ("ch19_linear_algebra_essentials_fig1.png",
         "Vector addition visualized: a, b, and a+b as arrows from the origin, with the angle between a and b labeled."),
        ("ch19_linear_algebra_essentials_fig2.png",
         "Eigenvectors of a covariance matrix, drawn as red arrows over the correlated data they describe -- pointing exactly along the data's natural axes of spread."),
    ],
    "20_multivariate_pca.md": [
        ("ch20_multivariate_pca_fig1.png",
         "Elliptical density contours of a Multivariate Normal distribution -- the ellipse's tilt and stretch are determined entirely by the covariance matrix."),
        ("ch20_multivariate_pca_fig2.png",
         "Principal component directions (red arrows) overlaid on correlated 2D data, with arrow length proportional to the variance each component explains."),
        ("ch20_multivariate_pca_fig3.png",
         "A scree plot on 10-feature data generated from only 3 true underlying factors: notice the sharp 'elbow' after the 3rd principal component."),
    ],
    "21_information_theory.md": [
        ("ch21_information_theory_fig1.png",
         "Entropy of a biased coin as a function of P(heads), maximized exactly at p=0.5 -- a fair coin is the most unpredictable possible two-outcome variable."),
    ],
    "22_statistical_learning_theory.md": [
        ("ch22_statistical_learning_theory_fig1.png",
         "Fitting PURE RANDOM NOISE (y has no real relationship to x) with increasingly flexible polynomials: training error keeps dropping while the true generalization error does not -- the overfitting gap widens with model capacity."),
    ],
    "23_latent_variable_models.md": [
        ("ch23_latent_variable_models_fig1.png",
         "Left: a single Normal distribution fit via plain MLE badly misrepresents bimodal data. Right: a 2-component Gaussian Mixture fit via EM correctly recovers the two true underlying clusters."),
    ],
    "24_markov_chains_monte_carlo.md": [
        ("ch24_markov_chains_monte_carlo_fig1.png",
         "The Sunny/Rainy Markov chain started from two different initial states, both converging to the same long-run stationary distribution (P(Sunny)=2/3) -- the chain 'forgets' where it started."),
        ("ch24_markov_chains_monte_carlo_fig2.png",
         "Samples drawn by a from-scratch Metropolis-Hastings sampler, histogrammed against the true target density -- recovered without ever computing the distribution's normalizing constant."),
    ],
    "25_statistics_in_neural_networks.md": [
        ("ch25_statistics_in_neural_networks_fig1.png",
         "The training loss curve of a tiny network -- this curve IS the negative log-likelihood, decreasing every epoch."),
        ("ch25_statistics_in_neural_networks_fig2.png",
         "A calibration / reliability diagram: predicted confidence vs actual accuracy. Points on the diagonal are perfectly calibrated."),
        ("ch25_statistics_in_neural_networks_fig3.png",
         "Activation magnitude through 10 layers under good, too-large, and too-small weight initialization -- showing exactly how poor scaling causes signal to explode or vanish with depth."),
    ],
}

inserted, skipped = 0, 0
for filename, figs in FIGURES.items():
    path = os.path.join(CHAPTERS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "## Figures" in content:
        print(f"SKIP (already has Figures section): {filename}")
        skipped += 1
        continue

    if "## Run it yourself" not in content:
        print(f"WARNING: no '## Run it yourself' heading found in {filename}, skipping")
        continue

    lines = ["## Figures", ""]
    for img, caption in figs:
        lines.append(f"![{caption}](../images/{img})")
        lines.append("")
        lines.append(f"*{caption}*")
        lines.append("")
    figures_block = "\n".join(lines) + "\n"

    new_content = content.replace("## Run it yourself", figures_block + "## Run it yourself", 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"INSERTED into: {filename}")
    inserted += 1

print(f"\nDone. Inserted into {inserted} files, skipped {skipped}.")
