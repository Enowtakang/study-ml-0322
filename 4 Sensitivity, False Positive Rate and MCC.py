"""
Computing:
            - Sensitivity,
            - Specificity,
            - False positive rate (FPR) and
            - Matthews correlation coefficient.

Abbreviations:
            TP - True Positive
            TN - True Negative
            FP - False Positive
            FN - False Negative

"""
from math import sqrt


# Sensitivity


def sensitivity(TP, TN, FP, FN):
    sensitivity_score = round(TP/(TP + FN), 3)
    print(f"Model Sensitivity = {sensitivity_score}")
    print("=========================")


# Specificity and FPR


def specificity_fpr(TP, TN, FP, FN):
    specificity_score = round(TN/(TN + FP), 3)
    fpr = round(1 - specificity_score, 3)
    print(f"Model Specificity = {specificity_score}")
    print("=========================")
    print(f"False Positive Rate = {fpr}")
    print("=========================")


# Matthews Correlation Coefficient


def matthews_correlation_coefficient(TP, TN, FP, FN):
    mcc = round(
        ((TP * TN) - (FP * FN))/sqrt(
            ((TP + FP) * (TP + FN) * (TN + FP) * (TN+FN))), 3)
    print(f"Matthews Correlation Coefficient = {mcc}")
    print(" ")
    print("END")


# Print all results


def all_results(TP, TN, FP, FN):
    sensitivity(TP, TN, FP, FN)
    specificity_fpr(TP, TN, FP, FN)
    matthews_correlation_coefficient(TP, TN, FP, FN)


# all_results()
