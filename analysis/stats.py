import os
import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.stats import ttest_rel, wilcoxon, shapiro, levene
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM

# load model results from /formatted_data
trial_types = ["solo", "solo_con", "pair", "pair_con"]
model_results = {}
TLX_results = {}
eye_tracking_results = {}
gsr_results = {}

for trial_type in trial_types:
    model_results[trial_type] = json.load(open(f"formatted_data/{trial_type}_model_results.json"))
    TLX_results[trial_type] = pd.read_csv(f"formatted_data/{trial_type}_TLX_results.csv").to_dict()
    eye_tracking_results[trial_type] = pd.read_csv(f"formatted_data/{trial_type}_eye_results.csv").to_dict()
    gsr_results[trial_type] = pd.read_csv(f"formatted_data/{trial_type}_gsr_results.csv").to_dict()

# Assuming `gsr_values` are read from the CSV and might not be in the correct format
# Convert each value to float, handling non-numeric values gracefully

def safe_float_convert(x):
    try:
        return float(x)
    except ValueError:
        return np.nan  # Convert non-numeric values to NaN


def t_test(data1, data2):
    print("Normality Check:")
    print("Con:", stats.shapiro(data1))
    print("No Con:", stats.shapiro(data2))

    if stats.shapiro(data1)[1] > 0.05 and stats.shapiro(data2)[1] > 0.05:
        print("Data is normally distributed, use parametric test")
        t_stat, p_value = stats.ttest_ind(data1, data2)
        print("T-Test:")
        print("T-statistic:", t_stat)
        print("P-value:", p_value)
    else:
        print("Data is not normally distributed, use non-parametric test")
        u_stat, p_value = stats.mannwhitneyu(data1, data2)
        print("U-Test:")
        print("U-statistic:", u_stat)
        print("P-value:", p_value)

def paired_test(data1, data2):
    print("Normality Check:")
    print("Data1:", stats.shapiro(data1))
    print("Data2:", stats.shapiro(data2))

    if stats.shapiro(data1)[1] > 0.05 and stats.shapiro(data2)[1] > 0.05:
        print("Data is normally distributed, use paired parametric test")
        t_stat, p_value = stats.ttest_rel(data1, data2)
        print("Paired T-Test:")
        print("T-statistic:", t_stat)
        print("P-value:", p_value)
    else:
        print("Data is not normally distributed, use non-parametric test")
        w_stat, p_value = stats.wilcoxon(data1, data2)
        print("Wilcoxon Signed-Rank Test:")
        print("W-statistic:", w_stat)
        print("P-value:", p_value)


"""## Research Question 1: How does pair-based collaboration impact cognitive load and model performance in an annotation task in an IML system?

- Dependent Variables: Cognitive load (Measured by: GSR, eye-tracker, NASA-TLX scores), LSTM Model performance (Measured by: Accuracy, Precision, Recall)
- Independent Variable: Collaboration mode (solo vs. pair)

### Pair-based collaboration impact on cognitive load

#### t-test on TLX Scores
"""

validation_category = "Rating Scale Score"
solo_TLX_scores = []
pair_TLX_scores = []

for category in TLX_results["solo"].keys():
    if category.startswith(validation_category):
        solo_TLX_scores.append(TLX_results["solo"][category][0])
for category in TLX_results["solo_con"]:
    if category.startswith(validation_category):
        solo_TLX_scores.append(TLX_results["solo_con"][category][0])

for category in TLX_results["pair"]:
    if category.startswith(validation_category):
        print(category)
        pair_TLX_scores.append(TLX_results["pair"][category][0])
for category in TLX_results["pair_con"]:
    if category.startswith(validation_category):
        pair_TLX_scores.append(TLX_results["pair_con"][category][0])

print(solo_TLX_scores)
print(pair_TLX_scores)

t_test(solo_TLX_scores, pair_TLX_scores)
paired_test(solo_TLX_scores, pair_TLX_scores)

"""#### t-test on Average Pupil Diameter"""

solo_average_diameter_results = []
pair_average_diameter_results = []

for category in eye_tracking_results["solo"].keys():
    if category.startswith("diameter"):
        diameters = list(eye_tracking_results["solo"][category].values())
        diameters = [diameter for diameter in diameters if not np.isnan(diameter)]
        average_diameter = np.mean(diameters)
        solo_average_diameter_results.append(average_diameter)
for category in eye_tracking_results["solo_con"]:
    if category.startswith("diameter"):
        diameters = list(eye_tracking_results["solo_con"][category].values())
        diameters = [diameter for diameter in diameters if not np.isnan(diameter)]
        average_diameter = np.mean(diameters)
        solo_average_diameter_results.append(average_diameter)

for category in eye_tracking_results["pair"]:
    if category.startswith("diameter"):
        diameters = list(eye_tracking_results["pair"][category].values())
        diameters = [diameter for diameter in diameters if not np.isnan(diameter)]
        average_diameter = np.mean(diameters)
        pair_average_diameter_results.append(average_diameter)
for category in eye_tracking_results["pair_con"]:
    if category.startswith("diameter"):
        diameters = list(eye_tracking_results["pair_con"][category].values())
        diameters = [diameter for diameter in diameters if not np.isnan(diameter)]
        average_diameter = np.mean(diameters)
        pair_average_diameter_results.append(average_diameter)

print(solo_average_diameter_results)
print(pair_average_diameter_results)

t_test(solo_average_diameter_results, pair_average_diameter_results)
paired_test(solo_average_diameter_results, pair_average_diameter_results)

"""#### t-test on Average Skin Conductance"""

solo_average_gsr = []
pair_average_gsr = []

for category in gsr_results["solo"].keys():
    if category.startswith("GSR Conductance CAL (u Siemens)"):
        gsr_values = list(gsr_results["solo"][category].values())
        gsr_values = [safe_float_convert(gsr) for gsr in gsr_values]
        gsr_values = [gsr for gsr in gsr_values if not np.isnan(gsr)]
        average_gsr = np.mean(gsr_values)
        solo_average_gsr.append(average_gsr)
for category in gsr_results["solo_con"].keys():
    if category.startswith("GSR Conductance CAL (u Siemens)"):
        gsr_values = list(gsr_results["solo_con"][category].values())
        gsr_values = [safe_float_convert(gsr) for gsr in gsr_values]
        gsr_values = [gsr for gsr in gsr_values if not np.isnan(gsr)]
        average_gsr = np.mean(gsr_values)
        solo_average_gsr.append(average_gsr)

for category in gsr_results["pair"].keys():
    if category.startswith("GSR Conductance CAL (u Siemens)"):
        gsr_values = list(gsr_results["pair"][category].values())
        gsr_values = [safe_float_convert(gsr) for gsr in gsr_values]
        gsr_values = [gsr for gsr in gsr_values if not np.isnan(gsr)]
        average_gsr = np.mean(gsr_values)
        pair_average_gsr.append(average_gsr)
for category in gsr_results["pair_con"].keys():
    if category.startswith("GSR Conductance CAL (u Siemens)"):
        gsr_values = list(gsr_results["pair_con"][category].values())
        gsr_values = [safe_float_convert(gsr) for gsr in gsr_values]
        gsr_values = [gsr for gsr in gsr_values if not np.isnan(gsr)]
        average_gsr = np.mean(gsr_values)
        pair_average_gsr.append(average_gsr)



print(solo_average_gsr)
print(pair_average_gsr)

t_test(solo_average_gsr, pair_average_gsr)
paired_test(solo_average_gsr, pair_average_gsr)

"""### ANOVA"""



"""### Pair-based collaboration impact on model performance

#### t-test on accuracy with respect to dataset
"""

validation_type = "Updated model accuracy with respect to dataset"
solo_validation_values = []
pair_validation_values = []

for category in model_results["solo"].keys():
    if category.startswith(validation_type):
        solo_validation_values.append(model_results["solo"][category])
for category in model_results["solo_con"]:
    if category.startswith(validation_type):
        solo_validation_values.append(model_results["solo_con"][category])

for category in model_results["pair"]:
    if category.startswith(validation_type):
        pair_validation_values.append(model_results["pair"][category])
for category in model_results["pair_con"]:
    if category.startswith(validation_type):
        pair_validation_values.append(model_results["pair_con"][category])

print(solo_validation_values)
print(pair_validation_values)

t_test(solo_validation_values, pair_validation_values)
paired_test(solo_validation_values, pair_validation_values)

"""## Research Question 2: What is the effect of varying the controllability of the system on cognitive load and model performance in an IML system?

- Dependent Variables: Cognitive load (Measured by: GSR, eye-tracker, NASA-TLX scores), LSTM Model performance (Measured by: Accuracy, Precision, Recall)
- Independent Variable: Controllability (low vs. high)

### Controllability of the system impact on cognitive load

#### t-test on TLX Scores
"""

validation_category = "Rating Scale Score"
con_tlx_scores = []
no_con_tlx_scores = []

for category in TLX_results["solo"].keys():
    if category.startswith(validation_category):
        no_con_tlx_scores.append(TLX_results["solo"][category][0])
for category in TLX_results["pair"]:
    if category.startswith(validation_category):
        no_con_tlx_scores.append(TLX_results["pair"][category][0])

for category in TLX_results["solo_con"]:
    if category.startswith(validation_category):
        con_tlx_scores.append(TLX_results["solo_con"][category][0])
for category in TLX_results["pair_con"]:
    if category.startswith(validation_category):
        con_tlx_scores.append(TLX_results["pair_con"][category][0])

print(con_tlx_scores)
print(no_con_tlx_scores)

t_test(con_tlx_scores, no_con_tlx_scores)
paired_test(con_tlx_scores, no_con_tlx_scores)

"""#### t-test on average diameter"""

con_validation_values = []
no_con_validation_values = []

for category in eye_tracking_results["solo"].keys():
    if category.startswith("diameter"):
        diameters = list(eye_tracking_results["solo"][category].values())
        diameters = [diameter for diameter in diameters if not np.isnan(diameter)]
        average_diameter = np.mean(diameters)
        no_con_validation_values.append(average_diameter)
for category in eye_tracking_results["pair"]:
    if category.startswith("diameter"):
        diameters = list(eye_tracking_results["pair"][category].values())
        diameters = [diameter for diameter in diameters if not np.isnan(diameter)]
        average_diameter = np.mean(diameters)
        no_con_validation_values.append(average_diameter)

for category in eye_tracking_results["solo_con"]:
    if category.startswith("diameter"):
        diameters = list(eye_tracking_results["solo_con"][category].values())
        diameters = [diameter for diameter in diameters if not np.isnan(diameter)]
        average_diameter = np.mean(diameters)
        con_validation_values.append(average_diameter)
for category in eye_tracking_results["pair_con"]:
    if category.startswith("diameter"):
        diameters = list(eye_tracking_results["pair_con"][category].values())
        diameters = [diameter for diameter in diameters if not np.isnan(diameter)]
        average_diameter = np.mean(diameters)
        con_validation_values.append(average_diameter)

print(con_validation_values)
print(no_con_validation_values)

t_test(con_validation_values, no_con_validation_values)
paired_test(con_validation_values, no_con_validation_values)

"""#### t-test on skin conductance"""

con_gsr = []
no_con_gsr = []

for category in gsr_results["solo"].keys():
    if category.startswith("GSR Conductance CAL (u Siemens)"):
        gsr_values = list(gsr_results["solo"][category].values())
        gsr_values = [safe_float_convert(gsr) for gsr in gsr_values]
        gsr_values = [gsr for gsr in gsr_values if not np.isnan(gsr)]
        average_gsr = np.mean(gsr_values)
        no_con_gsr.append(average_gsr)
for category in gsr_results["pair"]:
    if category.startswith("GSR Conductance CAL (u Siemens)"):
        gsr_values = list(gsr_results["pair"][category].values())
        gsr_values = [safe_float_convert(gsr) for gsr in gsr_values]
        gsr_values = [gsr for gsr in gsr_values if not np.isnan(gsr)]
        average_gsr = np.mean(gsr_values)
        no_con_gsr.append(average_gsr)

for category in gsr_results["solo_con"]:
    if category.startswith("GSR Conductance CAL (u Siemens)"):
        gsr_values = list(gsr_results["solo_con"][category].values())
        gsr_values = [safe_float_convert(gsr) for gsr in gsr_values]
        gsr_values = [gsr for gsr in gsr_values if not np.isnan(gsr)]
        average_gsr = np.mean(gsr_values)
        con_gsr.append(average_gsr)
for category in gsr_results["pair_con"]:
    if category.startswith("GSR Conductance CAL (u Siemens)"):
        gsr_values = list(gsr_results["pair_con"][category].values())
        gsr_values = [safe_float_convert(gsr) for gsr in gsr_values]
        gsr_values = [gsr for gsr in gsr_values if not np.isnan(gsr)]
        average_gsr = np.mean(gsr_values)
        con_gsr.append(average_gsr)

print(con_gsr)
print(no_con_gsr)

t_test(con_gsr, no_con_gsr)
paired_test(con_gsr, no_con_gsr)

"""### Controllability of the system impact on model performance"""

validation_type = "Updated model accuracy with respect to dataset"
con_validation_values = []
no_con_validation_values = []

for category in model_results["solo"].keys():
    if category.startswith(validation_type):
        no_con_validation_values.append(model_results["solo"][category])
for category in model_results["pair"]:
    if category.startswith(validation_type):
        no_con_validation_values.append(model_results["pair"][category])

for category in model_results["solo_con"]:
    if category.startswith(validation_type):
        con_validation_values.append(model_results["solo_con"][category])
for category in model_results["pair_con"]:
    if category.startswith(validation_type):
        con_validation_values.append(model_results["pair_con"][category])

print(con_validation_values)
print(no_con_validation_values)

t_test(con_validation_values, no_con_validation_values)
paired_test(con_validation_values, no_con_validation_values)



# Assuming `data` is a Pandas DataFrame containing 'subject_id', 'condition', and 'score'
# `subject_id`: Identifier for each subject
# `condition`: The condition or treatment (e.g., 'solo', 'pair')
# `score`: The dependent variable (e.g., TLX score, pupil diameter)

def run_repeated_measures_anova(data):
    anova = AnovaRM(data, depvar='score', subject='subject_id', within=['condition'])
    res = anova.fit()
    print(res.summary())



# Assuming `data` is a Pandas DataFrame containing 'subject_id', 'condition', and 'score'
# `subject_id`: Identifier for each subject
# `condition`: The condition or treatment (e.g., 'solo', 'pair')
# `score`: The dependent variable (e.g., TLX score, pupil diameter)

def run_repeated_measures_anova(data):
    anova = AnovaRM(data, depvar='score', subject='subject_id', within=['condition'])
    res = anova.fit()
    print(res.summary())


