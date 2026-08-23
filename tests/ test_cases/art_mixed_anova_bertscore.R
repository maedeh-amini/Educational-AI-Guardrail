# ==============================================================================
   ARTANOVA_ BERTSCORE 
# ==============================================================================

# 1: required packages 
# install.packages("ARTool")
# install.packages("dplyr")
# install.packages("emmeans")
# install.packages("ggplot2")
# install.packages("lme4") 
# install.packages("car") 
# install.packages("effectsize") 
# install.packages("ggdist")    

# 2: Load the primary packages
library(ARTool)
library(dplyr)
library(emmeans)
library(ggplot2)
library(lme4)
library(car)         # For assumption checks
library(effectsize)  # For effect sizes
library(ggdist)      # For Raincloud plots

# Step 3: Load the data
data_BERTScore <- read.csv(choose.files(), header=T)

# Convert categorical variables to factors
data_BERTScore$Architecture <- as.factor(data_BERTScore$Architecture)
data_BERTScore$DataType <- as.factor(data_BERTScore$DataType)
data_BERTScore$Question_ID <- as.factor(data_BERTScore$Question_ID)
str(data_BERTScore)

# ==============================================================================
# PHASE 1: EXPLORATORY DATA ANALYSIS (SCORE)
# ==============================================================================

print("--- PHASE 1: Descriptive Statistics for BERTScore ---")
# Calculate descriptive statistics using dplyr
score_summary <- data_BERTScore %>%
  group_by(Architecture, DataType) %>%
  dplyr::summarise(
    N = dplyr::n(),
    Mean = mean(Score, na.rm = TRUE),
    Median = median(Score, na.rm = TRUE),
    SD = sd(Score, na.rm = TRUE),
    Min = min(Score, na.rm = TRUE),
    Max = max(Score, na.rm = TRUE),
    .groups = 'drop'
  )
print(score_summary)

# Visualize the Score Data
# A. Histogram with Density Curves to visually check skewness
plot_score_hist_curve <- ggplot(data_BERTScore, aes(x = Score, fill = DataType, color = DataType)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 0.02, position = "identity", alpha = 0.4) +
  geom_density(alpha = 0.2, linewidth = 1) +
  facet_wrap(~ Architecture) +
  labs(title = "Distribution of BERTScore F1 with Density Curves", 
       x = "Score", 
       y = "Density") +
  theme_minimal()
print(plot_score_hist_curve)

# B. Boxplot to visually check for equal variance and outliers
plot_score_box <- ggplot(data_BERTScore, aes(x = Architecture, y = Score, fill = DataType)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Boxplot of BERTScore F1", x = "Architecture", y = "Score") +
  theme_minimal()
print(plot_score_box)


# ==============================================================================
# PHASE 2: ASSUMPTION CHECKS FOR ANOVA
# ==============================================================================

print("--- PHASE 2: ANOVA Assumption Checks ---")

# 1. Normality Check (Shapiro-Wilk Test on model residuals)
# We build a basic linear model first to extract the residuals
basic_lm <- lm(Score ~ Architecture * DataType, data = data_BERTScore)

# --- Q-Q Plot for Visual Normality Check ---
plot_qq <- ggplot(data.frame(residuals = residuals(basic_lm)), aes(sample = residuals)) +
  stat_qq() +
  stat_qq_line(color = "red", linewidth = 1) +
  labs(title = "Q-Q Plot of Model Residuals",
       x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_minimal()
print(plot_qq)
# ------------------------------------------------

shapiro_result <- shapiro.test(residuals(basic_lm))

print("1. Shapiro-Wilk Test for Normality:")
print(shapiro_result)
if(shapiro_result$p.value < 0.05) {
  print("-> CONCLUSION: Data violates normality assumption (p < 0.05). Non-parametric test recommended.")
} else {
  print("-> CONCLUSION: Data meets normality assumption (p > 0.05).")
}

# 2. Homogeneity of Variance Check (Levene's Test)
levene_result <- leveneTest(Score ~ Architecture * DataType, data = data_BERTScore)
print("2. Levene's Test for Homogeneity of Variance:")
print(levene_result)
if(levene_result$`Pr(>F)`[1] < 0.05) {
  print("-> CONCLUSION: Data violates equal variance assumption (heteroscedasticity detected).")
} else {
  print("-> CONCLUSION: Data meets equal variance assumption.")
}

print("FINAL DECISION: If either assumption is severely violated, proceed to Phase 3 (ART ANOVA).")
cat("\nPress [Enter] to continue to ART ANOVA and Error Rate Analysis...\n")
readline()


# ==============================================================================
# PHASE 3: NON-PARAMETRIC ANALYSIS (ART ANOVA)
# ==============================================================================

print("--- PHASE 3: Mixed ART ANOVA Results ---")
# Use the ARTool Package for Mixed ART ANOVA
art_model <- art(Score ~ Architecture * DataType + (1|Question_ID), data = data_BERTScore)
art_anova_results <- anova(art_model)
print(art_anova_results)

# Post-hoc Analysis 
# Note: Always running contrasts here to ensure the effect size block has data.
interaction_contrasts <- art.con(art_model, "Architecture:DataType", adjust = "tukey")
print("--- Tukey Post-Hoc Contrasts ---")
summary(interaction_contrasts)

# Visualize the final Score results
plot_score_bar <- ggplot(data_BERTScore, aes(x = Architecture, y = Score, fill = DataType)) +
  stat_summary(fun = mean, geom = "bar", position = "dodge") +
  labs(title = "Mean BERTScore F1 by Architecture and Data Type",
       x = "RAG Architecture", y = "Mean BERTScore F1") +
  theme_minimal()
print(plot_score_bar)

# ==============================================================================
# PHASE 3.5: EFFECT SIZES (CORRECTED FOR S4 OBJECTS)
# ==============================================================================

print("--- PHASE 3.5: Effect Sizes ---")
print("--- Partial Eta-Squared (ANOVA Effect Size) ---")
# Calculate Partial Eta-Squared using the F-value and degrees of freedom
eta_squared_results <- F_to_eta2(f = art_anova_results$F, 
                                 df = art_anova_results$Df, 
                                 df_error = art_anova_results$Df.res)

# Bind the names to the results for a clean table
eta_table <- data.frame(Effect = rownames(art_anova_results), eta_squared_results)
print(eta_table)


print("--- Hedges' g (Post-Hoc Effect Size) ---")
# Convert the S4 emmGrid object to a standard data frame
contrast_df <- as.data.frame(summary(interaction_contrasts))

# Take the Tukey contrast results and standardize the differences
hedges_g_results <- t_to_d(t = contrast_df$t.ratio, 
                           df = contrast_df$df)

# Convert Cohen's d to Hedges' g 
hedges_g_results$Hedges_g <- hedges_g_results$d * (1 - (3 / (4 * contrast_df$df - 1)))

# Bind to the contrast names for a clean table
g_table <- data.frame(Contrast = contrast_df$contrast, 
                      Hedges_g = hedges_g_results$Hedges_g)
print(g_table)

# ==============================================================================
# PHASE 4: ERROR RATE EXPLORATION & PUBLICATION VISUALS
# ==============================================================================

# 1. Create the Error_Rate column mathematically
data_BERTScore$Error_Rate <- 1 - data_BERTScore$Score

# 2. Advanced Summary: Calculate Standard Error and 95% CI for error bars
error_summary <- data_BERTScore %>%
  group_by(Architecture, DataType) %>%
  dplyr::summarise(
    N = dplyr::n(),
    Mean_Error = mean(Error_Rate, na.rm = TRUE),
    SD_Error = sd(Error_Rate, na.rm = TRUE),
    SE_Error = SD_Error / sqrt(N),
    CI_Error = qt(0.975, df = N - 1) * SE_Error, # 95% Confidence Interval multiplier
    Max_Error = max(Error_Rate, na.rm = TRUE),
    .groups = 'drop'
  )

print("--- PHASE 4: BERTScore Error Rate Summary with Confidence Intervals ---")
print(error_summary)

# 3. Histogram for Error Rates with Density Curves
plot_error_hist_curve <- ggplot(data_BERTScore, aes(x = Error_Rate, fill = DataType, color = DataType)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 0.02, position = "identity", alpha = 0.4) +
  geom_density(alpha = 0.2, linewidth = 1) +
  facet_wrap(~ Architecture) +
  labs(title = "Distribution of BERTScore Error Rates with Density Curves", 
       x = "Error Rate", 
       y = "Density") +
  theme_minimal()
print(plot_error_hist_curve)

# 4. Standard Box Plot for Error Rates
plot_error_box <- ggplot(data_BERTScore, aes(x = Architecture, y = Error_Rate, fill = DataType)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Box Plot of BERTScore Error Rates", x = "RAG Architecture", y = "Error Rate") +
  theme_minimal()
print(plot_error_box)

# 4b. NEW: Raincloud Plot (Half-Violin + Boxplot)
plot_raincloud <- ggplot(data_BERTScore, aes(x = Architecture, y = Error_Rate, fill = DataType)) +
  stat_halfeye(adjust = 0.5, width = 0.5, .width = 0, justification = -0.2, alpha = 0.6, position = position_dodge(width = 0.8)) +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.8, position = position_dodge(width = 0.8)) +
  scale_fill_viridis_d(option = "mako", end = 0.8) +
  labs(title = "BERTScore Error Rates: Raincloud Distribution",
       x = "RAG Architecture", y = "Error Rate") +
  theme_classic(base_size = 14) +
  theme(legend.position = "top")
print(plot_raincloud)

# 4c. NEW: Violin & Boxplot Combined Overlay
plot_violin_box <- ggplot(data_BERTScore, aes(x = Architecture, y = Error_Rate, fill = DataType)) +
  geom_violin(alpha = 0.5, trim = FALSE, position = position_dodge(width = 0.8)) +
  geom_boxplot(width = 0.2, color = "black", alpha = 0.9, position = position_dodge(width = 0.8)) +
  scale_fill_viridis_d(option = "mako", end = 0.8) +
  labs(title = "BERTScore Error Rates: Violin & Boxplot",
       x = "RAG Architecture", y = "Error Rate") +
  theme_classic(base_size = 14)
print(plot_violin_box)

# 5. NEW UPGRADED: Interaction Plot with 95% Confidence Interval Error Bars
pd <- position_dodge(0.1) # Dodging to prevent error bar overlap

plot_interaction_ci <- ggplot(error_summary, aes(x = DataType, y = Mean_Error, group = Architecture, color = Architecture)) +
  geom_line(linewidth = 1.2, position = pd) +
  geom_errorbar(aes(ymin = Mean_Error - CI_Error, ymax = Mean_Error + CI_Error), 
                width = 0.15, linewidth = 1, position = pd) +
  geom_point(size = 4, position = pd) +
  scale_color_viridis_d(option = "cividis", end = 0.8) +
  labs(title = "Interaction Effect: BERTScore Domain Shift",
       subtitle = "Error bars represent 95% Confidence Intervals",
       x = "Data Type", y = "Mean Error Rate") +
  theme_classic(base_size = 14)
print(plot_interaction_ci)


# ==============================================================================
# PHASE 5: SILENT STORAGE TO TARGET DIRECTORY
# ==============================================================================

output_dir <- "E:/THSI/Code/statistical_analyses _R/data and results/converted_data/bertscore"

# 1. Save data frames as CSV files
write.csv(score_summary, file.path(output_dir, "BERTScore_Score_Summary.csv"), row.names = FALSE)
write.csv(eta_table, file.path(output_dir, "BERTScore_ANOVA_EtaSquared.csv"), row.names = FALSE)
write.csv(g_table, file.path(output_dir, "BERTScore_Hedges_G.csv"), row.names = FALSE)
write.csv(error_summary, file.path(output_dir, "BERTScore_Error_Summary_CI.csv"), row.names = FALSE)

# 2. Sink raw statistical structures directly to the text file
sink(file.path(output_dir, "BERTScore_Full_Statistical_Report.txt"))
shapiro_result
levene_result
art_anova_results
summary(interaction_contrasts)
sink()

