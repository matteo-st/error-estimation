
from sklearn import metrics
import numpy as np

def fpr_at_fixed_tpr(fprs, tprs, thresholds, tpr_level: float = 0.95):
    
    idxs = [i for i, x in enumerate(tprs) if x >= tpr_level]
    if len(idxs) > 0:
        idx = min(idxs)
    else:
        idx = 0
    return fprs[idx], tprs[idx], thresholds[idx]

def auc_and_fpr_recall(conf, label, tpr_level: float = 0.95):
    # following convention in ML we treat OOD as positive


    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values

    fprs, tprs, thresholds = metrics.roc_curve(label, conf)
    fpr, tpr, thr = fpr_at_fixed_tpr(fprs, tprs, thresholds, tpr_level)

    auroc = metrics.auc(fprs, tprs)
    aupr_err = metrics.average_precision_score(label, conf)
    aupr_success = metrics.average_precision_score(1 - label, 1 - conf)

    return auroc, aupr_err, aupr_success, fpr, tpr, thr


def compute_all_metrics(conf, detector_labels):

    tpr_level = 0.95
    auroc, aupr_in, aupr_out, fpr, tpr, thr = auc_and_fpr_recall(conf, detector_labels, tpr_level)

    accuracy = np.mean(detector_labels)
    aurc_value = aurc(detector_labels, conf)

    return fpr, tpr, thr, auroc, accuracy, aurc_value, aupr_in, aupr_out

def rc_curve_stats(errors, conf) -> tuple[list[float], list[float], list[float]]:
        """
        Risk–Coverage curve computation.

        Adapted from:
        https://github.com/IML-DKFZ/fd-shifts
        (file: fd_shifts/analysis/metrics.py, function: rc_curve_stats)

        """
    
        coverages = []
        risks = []

        n_errors = len(errors)
        idx_sorted = np.argsort(conf)

        coverage = n_errors
        error_sum = sum(errors[idx_sorted])

        coverages.append(coverage / n_errors)
        risks.append(error_sum / n_errors)

        weights = []

        tmp_weight = 0
        for i in range(0, len(idx_sorted) - 1):
            coverage = coverage - 1
            error_sum = error_sum - errors[idx_sorted[i]]
            selective_risk = error_sum / (n_errors - 1 - i)
            tmp_weight += 1
            if i == 0 or conf[idx_sorted[i]] != conf[idx_sorted[i - 1]]:
                coverages.append(coverage / n_errors)
                risks.append(selective_risk)
                weights.append(tmp_weight / n_errors)
                tmp_weight = 0

        # add a well-defined final point to the RC-curve.
        if tmp_weight > 0:
            coverages.append(0)
            risks.append(risks[-1])
            weights.append(tmp_weight / n_errors)

        return coverages, risks, weights


def aurc(errors, conf) -> float:
    """AURC metric function
    Adapted from:
    https://github.com/IML-DKFZ/fd-shifts
    (file: fd_shifts/analysis/metrics.py, function: aurc)
    Args:
        errors: binary array indicating whether a prediction is incorrect (1) or correct (0)
        conf: confidence scores (higher means more confident)

    Returns:
        metric value
    """
    _, risks, weights = rc_curve_stats(errors, conf)
    aurc =  (
        sum(
            [
                (risks[i] + risks[i + 1]) * 0.5 * weights[i]
                for i in range(len(weights))
            ]
        )
    )

    return aurc