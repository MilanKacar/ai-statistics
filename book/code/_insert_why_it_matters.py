"""
Internal helper (not a chapter script): inserts a '## Why This Matters -- In
AI and In Everyday Life' section into each chapter, explicitly split into an
AI/ML bullet list and a non-AI real-life bullet list, right before
'## Check your understanding'. Idempotent: skips chapters that already have it.
"""
import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOK_DIR = os.path.dirname(CODE_DIR)
CHAPTERS_DIR = os.path.join(BOOK_DIR, "chapters")

# chapter_file -> (ai_bullets, reallife_bullets)
CONTENT = {
"01_data_populations_samples.md": (
    ["**Train/test splits are population vs. sample wearing a different hat**: your training set is a sample from the real-world population of inputs the model will eventually face; test accuracy is a *sample statistic* standing in for true population performance.",
     "**Dataset bias**: a face-recognition or hiring model trained on a non-representative sample of people will systematically misjudge the parts of the population it rarely saw -- a direct, high-stakes version of Chapter 1's population/sample gap."],
    ["**Political polling**: pollsters survey a few thousand people (the sample) to estimate how millions will vote (the population), and every reported result is an estimate with inherent uncertainty.",
     "**Factory quality control**: inspecting 50 items from a production run of 100,000 to estimate the overall defect rate, because checking every single unit isn't feasible."],
),
"02_descriptive_statistics.md": (
    ["**Latency monitoring**: production ML systems report p95/p99 latency (percentiles), not just the mean, because a skewed tail of slow requests is exactly what angry users experience and a mean would hide.",
     "**Outlier/label-noise detection**: spotting mislabeled training examples or corrupted sensor readings often starts with exactly this chapter's IQR-based outlier rule before any model is trained."],
    ["**Household income reporting**: governments report *median* household income, not mean, precisely because a few very wealthy households would otherwise distort the mean upward (Chapter 2's billionaire example, for real).",
     "**Personal budgeting**: looking at your median monthly spending versus the mean reveals whether one unusual expensive month is skewing your sense of \"normal\" spending."],
),
"03_probability_foundations.md": (
    ["**Combining independent model signals**: an ensemble or a spam filter combining several independent weak signals relies directly on the multiplication rule for independent events to compute a combined probability.",
     "**Calibrated classifier outputs**: a model saying \"93% cat\" is making a probability claim governed by exactly the axioms in this chapter -- it must behave like a real probability (between 0 and 1, consistent under AND/OR) to be trustworthy."],
    ["**Weather forecasts**: \"70% chance of rain\" is a real probability statement built from historical frequency (the empirical/frequentist approach from this chapter).",
     "**Everyday risk reasoning**: deciding whether to carry an umbrella by combining \"chance of rain\" with \"chance I'll be outside\" is literally the AND rule for independent-ish events."],
),
"04_bayes_theorem.md": (
    ["**Spam filters and Naive Bayes classifiers**: directly compute P(spam | words seen) from P(words | spam) and a prior spam rate -- this chapter's formula, applied word by word.",
     "**Bayesian A/B testing and Bayesian deep learning**: treat an unknown conversion rate or an unknown weight as a distribution to be updated, using exactly this posterior-equals-likelihood-times-prior recipe."],
    ["**Medical diagnosis**: doctors (should) combine a test's accuracy with how common a disease actually is in the patient's group -- exactly the worked example in this chapter, and a real source of diagnostic error when skipped.",
     "**Legal reasoning**: evaluating DNA or forensic evidence against how many people could plausibly match by chance (the \"prosecutor's fallacy\" mentioned above) is Bayes' Theorem misapplied in real courtrooms, with real consequences."],
),
"05_random_variables_discrete.md": (
    ["**A/B test conversion counts**: the number of conversions among n shown visitors is a Binomial random variable -- the generative model underneath every conversion-rate significance test.",
     "**Rare-event / anomaly monitoring**: modeling the number of fraud cases, server errors, or security incidents per day as Poisson is standard practice in production ML monitoring and anomaly detection."],
    ["**Call center staffing**: the number of customer calls arriving per hour is classically modeled as Poisson, directly driving how many staff to schedule.",
     "**Quality control sampling**: counting how many defective items turn up in a batch of n inspected units is a direct real-world Binomial calculation."],
),
"06_continuous_distributions.md": (
    ["**Neural network weight initialization and noise models**: weights are commonly drawn from Normal distributions, and regression/generative models routinely assume Normally-distributed noise -- both leaning directly on this chapter.",
     "**Sensor noise in robotics and autonomous vehicles**: measurement error from LIDAR, GPS, or cameras is frequently modeled as Normal or Exponential (for time-between-events) noise feeding into a perception pipeline."],
    ["**Standardized test scores**: SAT/IQ-style scores are explicitly constructed to be Normally distributed, and z-scores are how colleges and psychologists compare people across different tests.",
     "**Waiting for a bus or a bank queue**: the time between arrivals is a classic real-world Exponential-distribution scenario, memoryless property and all."],
),
"07_expectation_variance_covariance.md": (
    ["**Expected loss during training**: the quantity every ML training loop minimizes is literally an expected value (Chapter 7's E[X]) of a loss function over the training distribution.",
     "**Multicollinearity in ML features**: two highly correlated features (covariance near their product of standard deviations) carry redundant information, directly motivating PCA (Chapter 20) and regularization (Chapter 16)."],
    ["**Investment portfolio risk**: diversification is explicitly about picking assets with low or negative covariance, so losses in one don't compound with losses in another.",
     "**\"Regression to the mean\" in sports and business**: an athlete's amazing rookie season or a startup's unusually great quarter tends to be followed by something closer to average -- Galton's original discovery, still fooling people today."],
),
"08_central_limit_theorem.md": (
    ["**Trusting a validation accuracy number**: a larger validation/test set gives a tighter, more trustworthy estimate of true model performance, directly because of the CLT's shrinking standard error.",
     "**Ensembling**: averaging predictions from several semi-independent models reduces variance for the same $1/\\sqrt n$ reason the sample mean's spread shrinks -- the statistical justification behind bagging and random forests."],
    ["**Casinos**: a single hand of blackjack is wildly random, but a casino's profit *averaged* over millions of hands per year is remarkably stable and predictable -- the entire business model rests on the CLT.",
     "**Manufacturing batch averages**: the average weight of a 24-pack of cereal boxes is far more tightly controlled than any single box's weight, which is why quality control often checks batch averages."],
),
"09_sampling_and_estimation.md": (
    ["**Judging a reported model metric**: \"91% accuracy\" is an estimator's output with its own bias and variance -- this chapter is exactly what should make you ask what sample size and sampling procedure produced that number.",
     "**Biased training data as a biased estimator**: if your labeled training data was collected in a way that systematically over- or under-represents certain cases, your model's learned parameters inherit that bias, in the precise sense defined in this chapter."],
    ["**Choosing between average and median home price**: realtors and the median-vs-mean choice directly reflects which estimator (Chapter 2's robustness argument, Chapter 9's variance/bias tradeoff) better represents \"typical.\"",
     "**Government economic statistics**: unemployment rate and inflation figures are estimators computed from surveys, each with a known margin of error that government statisticians report alongside the headline number."],
),
"10_maximum_likelihood_estimation.md": (
    ["**Every standard neural network loss function**: MSE, binary cross-entropy, and categorical cross-entropy are all literally negative log-likelihoods under a specific distributional assumption -- this chapter is the direct derivation of how deep learning training objectives were chosen.",
     "**Logistic regression and most classical ML model fitting**: calling `.fit()` on a logistic regression or many other classical models is, under the hood, running exactly the MLE recipe from this chapter."],
    ["**Estimating a true defect rate from inspected samples**: if you inspect 50 items and find 3 defective, the \"obvious\" estimate of 6% defect rate is literally the MLE for a Binomial parameter, derivable by this chapter's recipe.",
     "**Fitting any theoretical curve to real measurements**: an actuary fitting a claim-size distribution to historical insurance claims, or a physicist fitting a decay curve to measurements, is running MLE."],
),
"11_confidence_intervals.md": (
    ["**Responsible model benchmarking**: reporting \"accuracy = 91.2% (95% CI: 89.8%-92.6%)\" instead of a bare number lets you tell whether a competitor's 92% is a real improvement or noise -- directly setting up Chapter 12.",
     "**Reporting uncertainty in any measured ML metric**: A/B test lift, fairness metrics, or latency improvements are all routinely reported with confidence intervals in serious ML engineering practice."],
    ["**Political polling margins of error**: \"52% support, margin of error +/-3%\" is a 95% confidence interval in everyday news reporting, built from exactly this chapter's formula.",
     "**Clinical trial results**: a drug's effect is reported as \"reduced risk by 15% (95% CI: 8%-22%)\" so doctors and regulators know how precisely the effect was actually pinned down."],
),
"12_hypothesis_testing.md": (
    ["**Deciding whether Model B actually beats Model A**: comparing a 1% accuracy gap between two models is exactly a hypothesis test in disguise -- without it, you can't tell a real improvement from test-set noise.",
     "**Feature/ablation significance in ML research**: papers routinely test whether removing a component (an ablation) causes a statistically significant performance drop before claiming the component matters."],
    ["**Website A/B testing**: \"does changing the button color actually change the click-through rate, or did we just get lucky with this batch of visitors\" is the canonical real-world use of this entire chapter.",
     "**Drug trials**: regulators require hypothesis tests (with pre-registered alpha and one/two-sided choice) before a new drug can be approved as genuinely effective versus placebo."],
),
"13_ttests_chisquare_anova.md": (
    ["**Comparing several model variants at once**: ANOVA (or its ML-flavored cousins) lets you test whether any of several model configurations differ in performance without the inflated false-positive risk of many separate t-tests.",
     "**Feature selection via chi-square**: testing whether a categorical feature is statistically associated with the target label (e.g. does device type relate to churn) is a standard chi-square test of independence used before model training."],
    ["**Fisher's original agricultural trials**: comparing crop yields across several fertilizer treatments at Rothamsted is literally what ANOVA was invented for, and the same logic still drives modern agricultural and industrial experiments.",
     "**Before/after medical measurements**: a paired t-test on patients' blood pressure before and after a treatment is standard practice in clinical research, exactly as in this chapter's worked example."],
),
"14_simple_linear_regression.md": (
    ["**The simplest baseline model in any ML pipeline**: before trying anything fancy, fitting simple linear regression tells you how much signal a single feature carries, and gives a sanity-check baseline to beat.",
     "**Interpretable feature effects**: a regression slope answering \"how much does predicted output change per unit of this one feature\" is often more valuable for stakeholder trust than a complex black-box model's equivalent behavior."],
    ["**Predicting house prices from square footage**: one of the most common real-estate analytics examples, and a direct application of this chapter's exact formula.",
     "**Predicting sales from advertising spend**: marketing teams routinely fit this exact simple regression to estimate return on ad spend before committing to a bigger campaign."],
),
"15_multiple_regression.md": (
    ["**Feature engineering pitfalls**: adding many engineered features to a model without checking adjusted R² or using cross-validation is a direct, common real cause of production models that looked great in development and failed in the real world.",
     "**Controlling for confounds in ML fairness analysis**: checking whether a model's predictions differ by a protected attribute *after* controlling for legitimate factors is a direct real application of multiple regression's \"holding other variables constant\" logic."],
    ["**Predicting salary from experience, education, and location together**: a classic HR analytics use of multiple regression, where the Simpson's Paradox risk from this chapter is a genuine, frequently-made mistake.",
     "**UC Berkeley admissions (Simpson's Paradox)**: the real 1973 case in this chapter is itself a profoundly real-life example of why controlling for confounders matters in any fairness-sensitive analysis, not just a toy illustration."],
),
"16_bias_variance_regularization.md": (
    ["**L1/L2 regularization, dropout, and cross-validation**: these are not separate tricks -- they are the standard, default tools in nearly every real ML training pipeline, and this chapter is their complete justification.",
     "**Hyperparameter tuning**: choosing a regularization strength, model depth, or number of training epochs via cross-validation is the single most common practical workflow in applied machine learning."],
    ["**A tailor fitting clothes**: too loose and it doesn't capture your shape (underfitting/high bias); tailored to one specific momentary body position and it won't move with you (overfitting/high variance) -- a genuinely useful everyday analogy for the tradeoff.",
     "**A student over-preparing for one specific practice exam**: memorizing the exact practice questions (low bias on that one test, but high variance -- it won't transfer) versus learning general principles (slightly imperfect on the practice test, but generalizes) is bias-variance in a classroom."],
),
"17_bayesian_statistics.md": (
    ["**Bayesian hyperparameter optimization**: tools used throughout ML research treat each hyperparameter setting's performance as uncertain and use Bayesian updating to decide what to try next, directly using this chapter's machinery.",
     "**Cold-start recommendation systems**: when a new user or item has almost no data, a sensible prior belief (informed by similar users/items) combined with the little data available is exactly Bayesian updating in production."],
    ["**Doctors updating a diagnosis as test results come in**: each new test result updates the posterior belief about what's wrong, which then becomes the prior for interpreting the next test -- real diagnostic reasoning, not just theory.",
     "**Weather forecasts updating through the day**: a morning forecast is a prior; afternoon radar data updates it to a posterior, which becomes the new prior for the evening forecast -- sequential Bayesian updating in daily life."],
),
"18_resampling_methods.md": (
    ["**Bootstrap confidence intervals for any ML metric**: when there's no clean textbook formula for a metric's uncertainty (e.g. an F1 score or a custom business metric), bootstrapping the test set is the standard practical fix used in real ML evaluation.",
     "**Permutation-based feature importance**: a standard, assumption-light way to measure how much a feature matters to a trained model is to permute (shuffle) that feature's values and see how much performance drops -- literally this chapter's permutation-test logic repurposed."],
    ["**Estimating uncertainty in a small-sample medical or industrial study**: when you only have a handful of measurements and no clean formula applies, bootstrapping lets you still get a defensible confidence interval.",
     "**Testing whether a process change actually helped**, using only the limited before/after data a small factory or clinic actually has, rather than assuming a textbook distribution that may not fit."],
),
"19_linear_algebra_essentials.md": (
    ["**Every single neural network forward pass**: a layer's computation is literally a matrix-vector multiplication (this chapter's core operation) followed by a nonlinearity -- there is no deep learning without this chapter.",
     "**Word/image embeddings**: representing words, images, or users as vectors, and measuring their similarity via the dot product/cosine similarity from this chapter, underlies search, recommendation, and language models alike."],
    ["**Rotating and scaling a photo**: every image-editing transformation (rotate, zoom, skew) is a matrix applied to the image's pixel coordinate vectors -- the \"matrix as a transformation of space\" idea from this chapter.",
     "**GPS triangulation**: computing your location from satellite signals is solved as a system of linear equations, the same normal-equations machinery used for regression in this chapter."],
),
"20_multivariate_pca.md": (
    ["**Dimensionality reduction before training**: compressing hundreds of correlated features (or high-dimensional embeddings) down to a handful of principal components is standard practice to fight the curse of dimensionality and speed up training.",
     "**Visualizing high-dimensional model embeddings**: projecting a language or vision model's internal representations down to 2-3 dimensions for visualization (often as a first step before more specialized tools) relies directly on this chapter's eigenvector machinery."],
    ["**Summarizing a company's dashboard**: reducing dozens of correlated KPIs (revenue, signups, churn, engagement) into a few \"principal\" health indicators that capture most of what's really moving is a direct real-world PCA use case.",
     "**Population genetics**: PCA on genetic marker data is a standard real technique for visualizing human population structure and ancestry, directly using this chapter's covariance-matrix eigen-decomposition."],
),
"21_information_theory.md": (
    ["**Cross-entropy loss, used in literally every standard classifier and every token prediction in a language model**: this chapter's entropy/cross-entropy formulas are not an analogy for deep learning loss functions -- they ARE the loss function.",
     "**KL divergence in VAEs, diffusion models, and RLHF**: keeping a fine-tuned language model's behavior close to its original pretrained behavior during reinforcement learning, or regularizing a generative model's latent space, is a direct real application of KL divergence."],
    ["**Data compression (zip files, JPEG, MP3)**: Shannon's original insight -- that the minimum bits needed to encode a message equals its entropy -- is the literal mathematical foundation every compression algorithm is built on.",
     "**Morse code's letter frequencies**: common letters like \"E\" get short codes and rare letters like \"Q\" get long codes, a real pre-Shannon intuition for exactly the entropy-minimizing coding this chapter formalizes."],
),
"22_statistical_learning_theory.md": (
    ["**Why deep learning's generalization is still an open research question**: this chapter's VC-dimension bounds, and where they fail to explain modern neural networks, is literally the subject of active, ongoing deep learning research.",
     "**Guiding model capacity and regularization choices**: understanding the generalization-gap tradeoff is what justifies practical choices like early stopping, weight decay, and architecture size limits in real ML systems."],
    ["**A student who aces practice tests but fails the real exam**: memorizing specific practice questions (fitting the \"training sample\" perfectly) without learning transferable principles is overfitting happening to a person, not a model.",
     "**Why a tailor-made process that works perfectly for one client often fails for the next**: a solution overly tuned to one specific case (high \"capacity\" relative to how much it was tested) often fails to generalize to new, similar-but-different cases."],
),
"23_latent_variable_models.md": (
    ["**Customer/user segmentation without labels**: clustering users into behavioral groups nobody explicitly labeled (a Gaussian Mixture Model or similar) is standard practice in product analytics and marketing ML.",
     "**Topic modeling and speaker clustering**: discovering hidden topics in a document collection (LDA, a close cousin of GMM) or clustering speakers in an audio recording both rely directly on this chapter's EM-style latent-variable fitting."],
    ["**Market segmentation**: a business grouping its customers into \"budget-conscious,\" \"premium,\" and \"occasional\" segments, without anyone having labeled customers that way in advance, is a real-world latent-variable clustering problem.",
     "**Discovering hidden subgroups in survey data**: social scientists routinely find that survey respondents cluster into unlabeled subgroups (e.g. political or cultural archetypes) using exactly this chapter's mixture-model logic."],
),
"24_markov_chains_monte_carlo.md": (
    ["**Google's original PageRank algorithm**: modeling a web surfer randomly clicking links as a Markov chain, and finding its stationary distribution, is literally how PageRank ranked the early web.",
     "**Reinforcement learning's entire mathematical foundation**: a Markov Decision Process (the formal object behind RL) is this chapter's Markov chain plus actions and rewards -- there is no modern RL theory without this chapter."],
    ["**Board games like Monopoly or snakes-and-ladders**: your next position depends only on your current position and the dice roll, not on how you got there -- a textbook Markov chain playing out at the kitchen table.",
     "**Queueing at a bank or call center**: models of how many people are in line next, given only how many are in line now (not the full history), use exactly this chapter's Markov chain machinery for staffing decisions."],
),
"25_statistics_in_neural_networks.md": (
    ["**This entire chapter IS the AI application**: softmax, weight initialization, batch normalization, weight decay, dropout, and training itself are all direct, named instances of the statistics covered across this whole book.",
     "**Model calibration in high-stakes deployments**: ensuring a medical-diagnosis or self-driving-car model's stated confidence actually matches its real accuracy (this chapter's reliability diagram) is a genuine, actively-managed real-world safety concern."],
    ["**A student learning from a teacher's corrections**: a loss function is mathematically a teacher's \"how wrong was that, and in which direction\" signal, and gradient descent is the student nudging their understanding step by step -- a real pedagogical analogy, not just a cute one.",
     "**Trusting an AI's stated confidence**: when a medical-imaging AI says \"85% confident this is malignant,\" whether that 85% is actually trustworthy (calibration) has direct, real consequences for patient care decisions made by real doctors."],
),
}


def build_section(ai_bullets, real_bullets):
    ai_lines = "\n".join(f"- {b}" for b in ai_bullets)
    real_lines = "\n".join(f"- {b}" for b in real_bullets)
    return (
        "## Why This Matters — In AI and In Everyday Life\n\n"
        "**In AI / machine learning:**\n\n"
        f"{ai_lines}\n\n"
        "**In everyday, non-AI life:**\n\n"
        f"{real_lines}\n\n"
    )


inserted, skipped = 0, 0
for filename, (ai_bullets, real_bullets) in CONTENT.items():
    path = os.path.join(CHAPTERS_DIR, filename)
    with open(path, encoding="utf-8") as f:
        content = f.read()

    if "Why This Matters" in content:
        print(f"SKIP (already present): {filename}")
        skipped += 1
        continue

    if "## Check your understanding" not in content:
        print(f"WARNING: no '## Check your understanding' heading in {filename}, skipping")
        continue

    section = build_section(ai_bullets, real_bullets)
    new_content = content.replace("## Check your understanding", section + "## Check your understanding", 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"INSERTED into: {filename}")
    inserted += 1

print(f"\nDone. Inserted into {inserted} files, skipped {skipped}.")
