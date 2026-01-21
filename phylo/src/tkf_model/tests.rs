use approx::assert_relative_eq;
use nalgebra::DVector;

use crate::alignment::{Alignment, AncestralAlignment, Mapping, Sequences, MASA};
use crate::alphabets::Alphabet;
use crate::likelihood::{
    ModelSearchCost, PARAM_RANGE_POSITIVE, PARAM_RANGE_UNIT_INTERVAL_EXCLUSIVE,
};
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{QMatrixMaker, SubstModel, SubstitutionCostBuilder as SCB};
use crate::substitution_models::{BLOSUM, GTR, HIVB, HKY, JC69, K80, TN93, WAG};
use crate::tkf_model::tkf92::TKF92IndelModel;
use crate::tkf_model::tkf_indel::DUMMY_FREQS;
use crate::tree::NodeIdx::{self, Internal, Leaf};
use crate::{frequencies, record_wo_desc as record, tree};

use super::*;

#[test]
fn tkf_dummy_freqs() {
    assert_eq!(&*DUMMY_FREQS, &DVector::<f64>::zeros(0));
}

#[cfg(test)]
pub(crate) fn get_mapping_for_any_node<'a, AA: AncestralAlignment>(
    msa: &'a AA,
    node: &'a NodeIdx,
) -> &'a Mapping {
    match node {
        Internal(_) => msa.ancestral_map(node),
        Leaf(_) => msa.leaf_map(node),
    }
}

// This is a direct implementation of the TKF91 log likelihood calculation without any
// aggregation over subtrees and without substitutions. This direct calculation is sufficient
// if one is only interested in the indel likelihood for a fixed alignment and tree.
// Used for testing purposes only, i.e., to validate the aggregated implementation.
#[cfg(test)]
fn tkf91_indel_logl_without_aggregation<AA: AncestralAlignment>(
    model: &TKF91IndelModel,
    phylo: &PhyloInfo<AA>,
) -> f64 {
    let tree = &phylo.tree;
    let lambda = model.lambda();
    let mu = model.mu();

    // for the root
    let mut prob: f64 = (1.0 - lambda / mu).ln();

    let mut last_event_deletion = vec![false; tree.len()];
    for i in 0..phylo.msa.len() {
        let mut event_prob = 1.0;
        if get_mapping_for_any_node(&phylo.msa, &phylo.tree.root)[i].is_some() {
            // the eq seq at the root has a fragment
            event_prob *= lambda / mu;
        }
        for node_idx in tree.postorder() {
            // skipping the root of the tree because it has no parent and therefore also no
            // mutations probabilities
            if node_idx == &tree.root {
                continue;
            }
            let node_id_value = usize::from(node_idx);

            let time = tree.node(node_idx).blen;
            let parent_id = &tree.node(node_idx).parent.unwrap();
            let parent_is_gap = get_mapping_for_any_node(&phylo.msa, parent_id)[i].is_none();
            let current_is_gap = get_mapping_for_any_node(&phylo.msa, node_idx)[i].is_none();

            let beta = beta(lambda, mu, time);
            if i == 0 {
                prob += log_i1(lambda, beta);
            }
            if parent_is_gap && current_is_gap {
                continue;
            } else if !parent_is_gap && !current_is_gap {
                // homolog block
                event_prob *= h1(lambda, mu, beta, time);
                last_event_deletion[node_id_value] = false;
            } else if !parent_is_gap && current_is_gap {
                // deletion
                event_prob *= n0(mu, beta);
                last_event_deletion[node_id_value] = true;
            } else if parent_is_gap && !current_is_gap {
                // insertion
                if last_event_deletion[node_id_value] {
                    prob += log_n1(lambda, mu, beta, time);
                    prob -= (lambda * beta).ln();
                    prob -= n0(mu, beta).ln();
                }
                event_prob *= lambda * beta;
                last_event_deletion[node_id_value] = false;
            }
        }
        prob += event_prob.ln();
    }
    prob
}

// This is a direct implementation of the TKF92 log likelihood calculation without any
// aggregation over subtrees and without substitutions. This direct calculation is sufficient
// if one is only interested in the indel likelihood for a fixed alignment and tree.
// Used for testing purposes only, i.e., to validate the aggregated implementation.
#[cfg(test)]
fn tkf92_indel_logl_without_aggregation<AA: AncestralAlignment>(
    model: &TKF92IndelModel,
    phylo: &PhyloInfo<AA>,
) -> f64 {
    let blocks = TKF92IndelModel::get_blocks(&phylo.msa);
    let tree = &phylo.tree;
    let lambda = model.lambda();
    let mu = model.mu();
    let r = model.params()[2];

    // for the root
    let mut prob: f64 = (1.0 - lambda / mu).ln();

    let mut last_event_deletion = vec![false; tree.len()];
    for (i, fragment) in blocks.iter().enumerate() {
        let mut event_prob = 1.0;
        let fragment_len = if i == 0 {
            *fragment
        } else {
            fragment - blocks[i - 1]
        };
        if get_mapping_for_any_node(&phylo.msa, &phylo.tree.root)[fragment - 1].is_some() {
            // the eq seq at the root has a fragment
            event_prob *= lambda / mu * (1.0 - r) / r;
            prob += fragment_len as f64 * r.ln();
        }
        for node_idx in tree.postorder() {
            // skipping the root of the tree because it has no parent and therefore also no
            // mutations probabilities
            if node_idx == &tree.root {
                continue;
            }
            let node_id_value = usize::from(node_idx);

            let time = tree.node(node_idx).blen;
            let parent_id = &tree.node(node_idx).parent.unwrap();
            let parent_is_gap =
                get_mapping_for_any_node(&phylo.msa, parent_id)[fragment - 1].is_none();
            let current_is_gap =
                get_mapping_for_any_node(&phylo.msa, node_idx)[fragment - 1].is_none();

            let beta = beta(lambda, mu, time);
            if i == 0 {
                prob += log_i1(lambda, beta);
            }
            if parent_is_gap && current_is_gap {
                continue;
            } else if !parent_is_gap && !current_is_gap {
                // homolog block
                event_prob *= h1(lambda, mu, beta, time);
                last_event_deletion[node_id_value] = false;
            } else if !parent_is_gap && current_is_gap {
                // deletion
                event_prob *= n0(mu, beta);
                last_event_deletion[node_id_value] = true;
            } else if parent_is_gap && !current_is_gap {
                // insertion
                if last_event_deletion[node_id_value] {
                    prob += log_n1(lambda, mu, beta, time);
                    prob -= (lambda * beta).ln();
                    prob -= n0(mu, beta).ln();
                }
                event_prob *= lambda * beta * (1.0 - r) / r;
                prob += fragment_len as f64 * r.ln();
                last_event_deletion[node_id_value] = false;
            }
        }
        prob += event_prob.ln();
        prob += (fragment_len - 1) as f64 * (1.0 + event_prob).ln();
    }
    prob
}

#[test]
fn tkf_beta() {
    assert_relative_eq!(beta(0.3, 0.5, 0.7), 0.5461782813185221);
}

#[test]
fn tkf_log_i1() {
    let l = 2.0;
    let m = 3.0;
    let time = 1.0;
    let b = beta(l, m, time);
    // log((1-2(1-e^(-1))/(3-2*e^(-1)))
    assert_relative_eq!(log_i1(l, b), -0.8172396554020775);
}

#[test]
fn tkf_log_n1() {
    let l = 2.0;
    let m = 3.0;
    let time = 0.5;
    let b = beta(l, m, time);
    // log((1-e^(-1.5) - 3(1-e^(-.5))/(3-2*e^(-.5)) )* (1-2(1-e^(-.5))/(3-2*e^(-.5)))   (2(1-e^(-1))/(3-2*e^(-1)))^0)
    assert_relative_eq!(log_n1(l, m, b, time), -2.732135332549935);
}

#[test]
fn tkf_n0() {
    let l = 2.0;
    let m = 3.0;
    let time = 0.5;
    let b = beta(l, m, time);
    // (3(1-e^(-.5))/(3-2*e^(-.5)))
    assert_relative_eq!(n0(m, b), 0.6605755607027574);
}

#[test]
fn tkf_h1() {
    let l = 2.0;
    let m = 3.0;
    let time = 1.5;
    let b = beta(l, m, time);
    // e^(-4.5) * (1-2(1-e^(-1.5))/(3-2*e^(-1.5)))
    assert_relative_eq!(h1(l, m, b, time), 0.004350089645603061);
}

#[test]
fn tkf_eta() {
    let l = 2.0;
    let m = 3.0;
    let time = 1.5;
    let b = beta(l, m, time);
    let n0 = n0(m, b);
    // math.log( (1 - math.exp(-3*1.5) - 3*((1 - math.exp((2-3)*1.5))/(3 - 2*math.exp((2-3)*1.5))))
    // * (1 - 2*((1 - math.exp((2-3)*1.5))/(3 - 2*math.exp((2-3)*1.5)))))
    // - math.log(2*((1 - math.exp((2-3)*1.5))/(3 - 2*math.exp((2-3)*1.5))))
    // - math.log(3*((1 - math.exp((2-3)*1.5))/(3 - 2*math.exp((2-3)*1.5))))
    assert_relative_eq!(eta(l, m, b, n0, time), -2.922778333826742);
}

#[test]
fn tkf91_get_blocks() {
    let tree = tree!("((A0:1.0,B1:1.0)I1:1.0);");
    let seqs = Sequences::new(vec![
        record!("A0", b"AAAB-D"),
        record!("B1", b"--ARAW"),
        record!("I1", b"AAAA-A"),
    ]);
    let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();

    let blocks = TKF91IndelModel::get_blocks(&msa);
    let block_lens = get_block_lengths(&blocks);

    assert_eq!(blocks, (1..msa.len() + 1).collect::<Vec<usize>>());
    assert_eq!(block_lens, vec![1; 6]);
}

#[test]
fn tkf92_get_blocks() {
    let tree = tree!("((A0:1.0,B1:1.0)I1:1.0);");
    let seqs = Sequences::new(vec![
        record!("A0", b"AAB-D"),
        record!("B1", b"-ARAW"),
        record!("I1", b"AAA-A"),
    ]);

    let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();

    let blocks = TKF92IndelModel::get_blocks(&msa);
    let block_lens = get_block_lengths(&blocks);

    assert_eq!(blocks, vec![1, 3, 4, 5]);
    assert_eq!(block_lens, vec![1, 2, 1, 1]);
}

#[cfg(test)]
pub(super) fn setup_test_phylo(alphabet: &'static Alphabet) -> PhyloInfo<MASA> {
    let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let msa = MASA::from_aligned_with_ancestral(
        Sequences::with_alphabet(
            vec![
                record!("A1", b"--GTGGA---"),
                record!("B2", b"-------NNA"),
                record!("I3", b"--T-------"),
                record!("C4", b"AGG-------"),
                record!("R5", b"--A-------"),
            ],
            alphabet,
        ),
        &tree,
    )
    .unwrap();
    PhyloInfo { msa, tree }
}

#[test]
fn tkf_indel_get_and_set_params_and_freqs() {
    let mut tkf_indel_cost =
        TKF92IndelCostBuilder::new(1.0, 2.0, 0.3, setup_test_phylo(Alphabet::dna()))
            .build()
            .unwrap();
    // params
    assert_eq!(tkf_indel_cost.param_count(), 3);
    assert_eq!(tkf_indel_cost.param(0), 1.0);
    assert_eq!(tkf_indel_cost.param(1), 2.0);
    assert_eq!(tkf_indel_cost.param(2), 0.3);
    assert_eq!(tkf_indel_cost.model.lambda(), 1.0);
    assert_eq!(tkf_indel_cost.model.mu(), 2.0);
    assert_eq!(tkf_indel_cost.model.r(), 0.3);
    tkf_indel_cost.set_param(2, 0.33);
    assert_eq!(tkf_indel_cost.param_count(), 3);
    assert_eq!(tkf_indel_cost.model.lambda(), 1.0);
    assert_eq!(tkf_indel_cost.model.mu(), 2.0);
    assert_eq!(tkf_indel_cost.model.r(), 0.33);
    // freqs
    assert_eq!(tkf_indel_cost.freqs(), &*DUMMY_FREQS);
    assert_eq!(
        tkf_indel_cost.empirical_freqs(),
        setup_test_phylo(Alphabet::dna()).freqs()
    );
}

#[test]
fn tkf_get_and_set_params() {
    let subst_model = SubstModel::<GTR>::new(&[0.1, 0.2, 0.3, 0.4], &[0.5, 0.6, 0.7, 0.8, 0.9]);
    let mut tkf_cost = TKF92CostBuilder::new(
        1.0,
        2.0,
        0.3,
        subst_model,
        setup_test_phylo(Alphabet::dna()),
    )
    .build()
    .unwrap();
    assert_eq!(tkf_cost.param_count(), 8);
    assert_eq!(tkf_cost.param(0), 1.0);
    assert_eq!(tkf_cost.param(1), 2.0);
    assert_eq!(tkf_cost.param(2), 0.3);
    assert_eq!(tkf_cost.param(3), 0.5);
    assert_eq!(tkf_cost.param(4), 0.6);
    assert_eq!(tkf_cost.param(5), 0.7);
    assert_eq!(tkf_cost.param(6), 0.8);
    assert_eq!(tkf_cost.param(7), 0.9);
    assert_eq!(tkf_cost.indel_cost.model.lambda(), 1.0);
    assert_eq!(tkf_cost.indel_cost.model.mu(), 2.0);
    assert_eq!(tkf_cost.indel_cost.model.r(), 0.3);
    tkf_cost.set_param(2, 0.33);
    tkf_cost.set_param(5, 0.77);
    assert_eq!(tkf_cost.param_count(), 8);
    assert_eq!(tkf_cost.param(0), 1.0);
    assert_eq!(tkf_cost.param(1), 2.0);
    assert_eq!(tkf_cost.param(2), 0.33);
    assert_eq!(tkf_cost.param(3), 0.5);
    assert_eq!(tkf_cost.param(4), 0.6);
    assert_eq!(tkf_cost.param(5), 0.77);
    assert_eq!(tkf_cost.param(6), 0.8);
    assert_eq!(tkf_cost.param(7), 0.9);

    assert_eq!(
        tkf_cost.empirical_freqs(),
        setup_test_phylo(Alphabet::dna()).freqs()
    );
}

#[test]
fn tkf91_indel_cost_fmt() {
    let tkf_indel_cost = TKF91IndelCostBuilder::new(1.0, 2.0, setup_test_phylo(Alphabet::dna()))
        .build()
        .unwrap();

    let fmt = format!("{}", tkf_indel_cost);

    assert_eq!(fmt, "TKF91 with lambda = 1, mu = 2");
}

#[test]
fn tkf91_cost_fmt() {
    let subst_model = SubstModel::<JC69>::new(&[], &[]);
    let tkf_cost = TKF91CostBuilder::new(1.0, 2.0, subst_model, setup_test_phylo(Alphabet::dna()))
        .build()
        .unwrap();

    let fmt = format!("{}", tkf_cost);

    assert_eq!(fmt, "TKF91 with lambda = 1, mu = 2 and JC69");
}

#[test]
fn tkf92_indel_cost_fmt() {
    let tkf_indel_cost =
        TKF92IndelCostBuilder::new(1.0, 2.0, 0.3, setup_test_phylo(Alphabet::dna()))
            .build()
            .unwrap();

    let fmt = format!("{}", tkf_indel_cost);

    assert_eq!(fmt, "TKF92 with lambda = 1, mu = 2, r = 0.3");
}

#[test]
fn tkf92_cost_fmt() {
    let subst_model = SubstModel::<JC69>::new(&[], &[]);
    let tkf_cost = TKF92CostBuilder::new(
        1.0,
        2.0,
        0.3,
        subst_model,
        setup_test_phylo(Alphabet::dna()),
    )
    .build()
    .unwrap();

    let fmt = format!("{}", tkf_cost);

    assert_eq!(fmt, "TKF92 with lambda = 1, mu = 2, r = 0.3 and JC69");
}

#[test]
fn tkf_get_and_set_freqs() {
    let subst_model = SubstModel::<GTR>::new(&[0.1, 0.2, 0.3, 0.4], &[0.5, 0.6, 0.7, 0.8, 0.9]);
    let mut tkf_cost = TKF92CostBuilder::new(
        1.0,
        2.0,
        0.3,
        subst_model,
        setup_test_phylo(Alphabet::dna()),
    )
    .build()
    .unwrap();
    assert_eq!(tkf_cost.freqs().as_slice(), &[0.1, 0.2, 0.3, 0.4]);
    tkf_cost.set_freqs(frequencies!(&[0.4, 0.3, 0.2, 0.1]));
    assert_eq!(tkf_cost.freqs().as_slice(), &[0.4, 0.3, 0.2, 0.1]);
}

#[test]
fn tkf91_param_range() {
    let subst_model = SubstModel::<GTR>::new(&[], &[]);
    let tkf_cost = TKF91CostBuilder::new(1.0, 2.0, subst_model, setup_test_phylo(Alphabet::dna()))
        .build()
        .unwrap();
    let lambda_range = tkf_cost.param_range(usize::from(TKF92Parameters::Lambda));
    let true_lambda_range = (f64::EPSILON, 2.0 - f64::EPSILON);
    assert_eq!(lambda_range, true_lambda_range);
    let mu_range = tkf_cost.param_range(usize::from(TKF92Parameters::Mu));
    let true_mu_range = (1.0 + f64::EPSILON, f64::MAX);
    assert_eq!(mu_range, true_mu_range);

    for subst_param_idx in 2..tkf_cost.param_count() {
        let subst_range = tkf_cost.param_range(subst_param_idx);
        let true_subst_range = PARAM_RANGE_POSITIVE;
        assert_eq!(subst_range, true_subst_range);
    }
}

#[test]
fn tkf92_param_range() {
    let subst_model = SubstModel::<GTR>::new(&[], &[]);
    let tkf_cost = TKF92CostBuilder::new(
        1.0,
        2.0,
        0.3,
        subst_model,
        setup_test_phylo(Alphabet::dna()),
    )
    .build()
    .unwrap();
    let lambda_range = tkf_cost.param_range(usize::from(TKF92Parameters::Lambda));
    let true_lambda_range = (f64::EPSILON, 2.0 - f64::EPSILON);
    assert_eq!(lambda_range, true_lambda_range);
    let mu_range = tkf_cost.param_range(usize::from(TKF92Parameters::Mu));
    let true_mu_range = (1.0 + f64::EPSILON, f64::MAX);
    assert_eq!(mu_range, true_mu_range);
    let r_range = tkf_cost.param_range(usize::from(TKF92Parameters::R));
    let true_r_range = PARAM_RANGE_UNIT_INTERVAL_EXCLUSIVE;
    assert_eq!(r_range, true_r_range);

    for subst_param_idx in 3..tkf_cost.param_count() {
        let subst_range = tkf_cost.param_range(subst_param_idx);
        let true_subst_range = PARAM_RANGE_POSITIVE;
        assert_eq!(subst_range, true_subst_range);
    }
}

#[test]
fn tkf91_indel_logl() {
    // arrange
    let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let seqs = Sequences::new(vec![
        record!("A1", b"--NNNNN---"),
        record!("B2", b"-------NNN"),
        record!("I3", b"--N-------"),
        record!("C4", b"NNN-------"),
        record!("R5", b"--N-------"),
    ]);
    let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();
    let phylo = PhyloInfo {
        msa,
        tree: tree.clone(),
    };
    let lambda = 0.1;
    let mu = 0.2;
    let tkf91_cost = TKF91IndelCostBuilder::new(lambda, mu, phylo)
        .build()
        .unwrap();

    // act
    let logl = tkf91_cost.logl();
    let half_manual = tkf91_indel_logl_without_aggregation(&tkf91_cost.model, &tkf91_cost.phylo);
    let mut manual_calculation = 0.0;
    manual_calculation += (1.0 - lambda / mu).ln();
    // immortal links
    manual_calculation += log_i1(lambda, beta(lambda, mu, tree.by_id("A1").blen));
    manual_calculation += log_i1(lambda, beta(lambda, mu, tree.by_id("B2").blen));
    manual_calculation += log_i1(lambda, beta(lambda, mu, tree.by_id("I3").blen));
    manual_calculation += log_i1(lambda, beta(lambda, mu, tree.by_id("C4").blen));
    // first block ([0:2], insertion at C4)
    let x = lambda * beta(lambda, mu, tree.by_id("C4").blen);
    manual_calculation += x.ln() * 2.0;
    // second block ([2:3], all homologous except B2 deleted)
    let mut x = lambda / mu;
    x *= h1(
        lambda,
        mu,
        beta(lambda, mu, tree.by_id("C4").blen),
        tree.by_id("C4").blen,
    );
    x *= h1(
        lambda,
        mu,
        beta(lambda, mu, tree.by_id("A1").blen),
        tree.by_id("A1").blen,
    );
    x *= h1(
        lambda,
        mu,
        beta(lambda, mu, tree.by_id("I3").blen),
        tree.by_id("I3").blen,
    );
    x *= n0(mu, beta(lambda, mu, tree.by_id("B2").blen));
    manual_calculation += x.ln();
    // third block ([3:7], insertion at A1)
    let x = lambda * beta(lambda, mu, tree.by_id("C4").blen);
    manual_calculation += x.ln() * 4.0;
    // fourth block ([7:10], insertion at B2)
    let x = lambda * beta(lambda, mu, tree.by_id("B2").blen);
    manual_calculation += x.ln() * 3.0;
    manual_calculation += log_n1(
        lambda,
        mu,
        beta(lambda, mu, tree.by_id("B2").blen),
        tree.by_id("B2").blen,
    );
    manual_calculation -= n0(mu, beta(lambda, mu, tree.by_id("B2").blen)).ln();
    manual_calculation -= (lambda * beta(lambda, mu, tree.by_id("B2").blen)).ln();

    // assert
    assert_relative_eq!(logl, manual_calculation);
    assert_relative_eq!(logl, half_manual);
}

#[test]
fn tkf92_indel_logl() {
    // arrange
    let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let seqs = Sequences::new(vec![
        record!("A1", b"--NNNNN---"),
        record!("B2", b"-------NNN"),
        record!("I3", b"--N-------"),
        record!("C4", b"NNN-------"),
        record!("R5", b"--N-------"),
    ]);
    let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();
    let m = msa.len() as f64;
    let phylo = PhyloInfo {
        msa,
        tree: tree.clone(),
    };
    let lambda = 0.1;
    let mu = 0.2;
    let r = 0.3;
    let tkf92_cost = TKF92IndelCostBuilder::new(lambda, mu, r, phylo)
        .build()
        .unwrap();

    // act
    let logl = tkf92_cost.logl();
    let half_manual = tkf92_indel_logl_without_aggregation(&tkf92_cost.model, &tkf92_cost.phylo);
    let mut manual_calculation = 0.0;
    manual_calculation += (1.0 - lambda / mu).ln();
    manual_calculation += m * r.ln();
    // immortal links
    manual_calculation += log_i1(lambda, beta(lambda, mu, tree.by_id("A1").blen));
    manual_calculation += log_i1(lambda, beta(lambda, mu, tree.by_id("B2").blen));
    manual_calculation += log_i1(lambda, beta(lambda, mu, tree.by_id("I3").blen));
    manual_calculation += log_i1(lambda, beta(lambda, mu, tree.by_id("C4").blen));
    // first block ([0:2], insertion at C4)
    let x = lambda * beta(lambda, mu, tree.by_id("C4").blen) * (1.0 - r) / r;
    manual_calculation += x.ln() + 1.0 * (1.0 + x).ln();
    // second block ([2:3], all homologous except B2 deleted)
    let mut x = lambda / mu * (1.0 - r) / r;
    x *= h1(
        lambda,
        mu,
        beta(lambda, mu, tree.by_id("C4").blen),
        tree.by_id("C4").blen,
    );
    x *= h1(
        lambda,
        mu,
        beta(lambda, mu, tree.by_id("A1").blen),
        tree.by_id("A1").blen,
    );
    x *= h1(
        lambda,
        mu,
        beta(lambda, mu, tree.by_id("I3").blen),
        tree.by_id("I3").blen,
    );
    x *= n0(mu, beta(lambda, mu, tree.by_id("B2").blen));
    manual_calculation += x.ln();
    // third block ([3:7], insertion at A1)
    let x = lambda * beta(lambda, mu, tree.by_id("C4").blen) * (1.0 - r) / r;
    manual_calculation += x.ln() + 3.0 * (1.0 + x).ln();
    // fourth block ([7:10], insertion at B2)
    let x = lambda * beta(lambda, mu, tree.by_id("B2").blen) * (1.0 - r) / r;
    manual_calculation += x.ln() + 2.0 * (1.0 + x).ln();
    manual_calculation += log_n1(
        lambda,
        mu,
        beta(lambda, mu, tree.by_id("B2").blen),
        tree.by_id("B2").blen,
    );
    manual_calculation -= n0(mu, beta(lambda, mu, tree.by_id("B2").blen)).ln();
    manual_calculation -= (lambda * beta(lambda, mu, tree.by_id("B2").blen)).ln();

    // assert
    assert_relative_eq!(logl, manual_calculation);
    assert_relative_eq!(logl, half_manual);
}

#[test]
fn tkf91_cost_builder_fails() {
    let phylo = setup_test_phylo(Alphabet::protein());
    let subst_model = SubstModel::<GTR>::new(&[], &[]);

    let tkf91_err_msg = TKF91CostBuilder::new(0.1, 0.2, subst_model, phylo)
        .build()
        .unwrap_err()
        .to_string();

    assert_eq!(
        tkf91_err_msg,
        "Alphabet mismatch between model and alignment"
    );
}

#[test]
fn tkf92_cost_builder_fails() {
    let phylo = setup_test_phylo(Alphabet::protein());
    let subst_model = SubstModel::<GTR>::new(&[], &[]);

    let tkf92_err_msg = TKF92CostBuilder::new(0.1, 0.2, 0.3, subst_model, phylo)
        .build()
        .unwrap_err()
        .to_string();

    assert_eq!(
        tkf92_err_msg,
        "Alphabet mismatch between model and alignment"
    );
}

#[test]
fn tkf91_logl_with_substitution() {
    // arrange
    let phylo = setup_test_phylo(Alphabet::dna());
    let subst_model = SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[1.2, 0.5, 5.0, 1.0, 1.0]);
    let subst_cost = SCB::new(subst_model.clone(), phylo.clone())
        .build()
        .unwrap();
    let lambda = 0.1;
    let mu = 0.2;
    let tkf_cost = TKF91CostBuilder::new(lambda, mu, subst_model, phylo)
        .build()
        .unwrap();

    // act
    let logl = tkf_cost.cost();
    let subst_logl = subst_cost.cost();
    let tkf_logl = tkf91_indel_logl_without_aggregation(
        &tkf_cost.indel_cost.model,
        &tkf_cost.indel_cost.phylo,
    );

    // assert
    assert_relative_eq!(logl, subst_logl + tkf_logl);
}

#[test]
fn tkf92_logl_with_substitution() {
    // arrange
    let phylo = setup_test_phylo(Alphabet::dna());
    let subst_model = SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[1.2, 0.5, 5.0, 1.0, 1.0]);
    let subst_cost = SCB::new(subst_model.clone(), phylo.clone())
        .build()
        .unwrap();
    let lambda = 0.1;
    let mu = 0.2;
    let r = 0.3;
    let tkf_cost = TKF92CostBuilder::new(lambda, mu, r, subst_model, phylo)
        .build()
        .unwrap();

    // act
    let logl = tkf_cost.cost();
    let subst_logl = subst_cost.cost();
    let tkf_logl = tkf92_indel_logl_without_aggregation(
        &tkf_cost.indel_cost.model,
        &tkf_cost.indel_cost.phylo,
    );

    // assert
    assert_relative_eq!(logl, subst_logl + tkf_logl, epsilon = 1e-12);
}

#[test]
fn tkf_indel_history_doesnt_change_felsenstein() {
    // arrange
    let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let seqs = Sequences::new(vec![
        record!("A1", b"--GTGTA---"),
        record!("B2", b"-------AGT"),
        record!("I3", b"--N-------"),
        record!("C4", b"GTA-------"),
        record!("R5", b"--N-------"),
    ]);
    let seqs2 = Sequences::new(vec![
        record!("A1", b"--GTGTA---"),
        record!("B2", b"-------AGT"),
        record!("I3", b"--NNNNNNNN"),
        record!("C4", b"GTA-------"),
        record!("R5", b"--NNNNN---"),
    ]);
    let msa1 = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();
    let msa2 = MASA::from_aligned_with_ancestral(seqs2, &tree).unwrap();
    let phylo1 = PhyloInfo {
        msa: msa1,
        tree: tree.clone(),
    };
    let phylo2 = PhyloInfo { msa: msa2, tree };
    let lambda = 0.1;
    let mu = 0.2;
    let r = 0.3;
    let subst_model = SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[1.2, 0.5, 5.0, 1.0, 1.0]);
    let tkf_cost1 = TKF92CostBuilder::new(lambda, mu, r, subst_model.clone(), phylo1)
        .build()
        .unwrap();

    let tkf_cost2 = TKF92CostBuilder::new(lambda, mu, r, subst_model, phylo2)
        .build()
        .unwrap();

    // act
    let tkf_log_1 = tkf_cost1.clone().cost();
    let tkf_log_2 = tkf_cost2.clone().cost();
    let tkf_indel_cost_1 = tkf_cost1.indel_cost.cost();
    let tkf_indel_cost_without_agg_1 = tkf92_indel_logl_without_aggregation(
        &tkf_cost1.indel_cost.model,
        &tkf_cost1.indel_cost.phylo,
    );
    let tkf_indel_cost_2 = tkf_cost2.indel_cost.cost();
    let tkf_indel_cost_without_agg_2 = tkf92_indel_logl_without_aggregation(
        &tkf_cost2.indel_cost.model,
        &tkf_cost2.indel_cost.phylo,
    );

    // assert
    assert_relative_eq!(tkf_indel_cost_1, tkf_indel_cost_without_agg_1);
    assert_relative_eq!(tkf_indel_cost_2, tkf_indel_cost_without_agg_2);
    assert_relative_eq!(tkf_log_1 - tkf_indel_cost_1, tkf_log_2 - tkf_indel_cost_2);
}

#[cfg(test)]
fn modify_tkf92_subst_params_costs_match_template<Q: QMatrix + QMatrixMaker>() {
    let phylo = setup_test_phylo(Q::alphabet());
    let subst_original_param = 1.0;
    let subst_changed_param = 0.5;
    let subst_model = SubstModel::<Q>::new(&[], &[subst_original_param]);
    let mut tkf_cost = TKF92CostBuilder::new(0.1, 0.2, 0.3, subst_model, phylo.clone())
        .build()
        .unwrap();

    // sanity check
    let logl = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(logl, ModelSearchCost::cost(&tkf_cost));

    // The likelihood should change if we change model parameters
    tkf_cost.set_param(3, subst_changed_param);
    let logl2 = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(logl2, ModelSearchCost::cost(&tkf_cost));
    assert_ne!(logl, logl2);

    // The likelihood should be the same if we rebuild from scratch with the same modification
    let subst_model = SubstModel::<Q>::new(&[], &[subst_changed_param]);
    let tkf_cost = TKF92CostBuilder::new(0.1, 0.2, 0.3, subst_model, phylo)
        .build()
        .unwrap();
    let new_logl = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(new_logl, ModelSearchCost::cost(&tkf_cost));
    assert_eq!(logl2, new_logl);
}

#[cfg(test)]
fn modify_tkf92_indel_params_costs_match_template<Q: QMatrix + QMatrixMaker>() {
    let phylo = setup_test_phylo(Q::alphabet());
    let subst_model = SubstModel::<Q>::new(&[], &[]);
    let tkf_original_mu = 0.2;
    let tkf_changed_mu = 0.25;
    let mut tkf_cost = TKF92CostBuilder::new(0.1, tkf_original_mu, 0.3, subst_model, phylo.clone())
        .build()
        .unwrap();

    // sanity check
    let logl = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(logl, ModelSearchCost::cost(&tkf_cost));

    // The likelihood should change if we change model parameters
    tkf_cost.set_param(1, tkf_changed_mu);
    let logl2 = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(logl2, ModelSearchCost::cost(&tkf_cost));
    assert_ne!(logl, logl2);

    // The likelihood should be the same if we rebuild from scratch with the same modification
    let subst_model = SubstModel::<Q>::new(&[], &[]);
    let tkf_cost = TKF92CostBuilder::new(0.1, tkf_changed_mu, 0.3, subst_model, phylo)
        .build()
        .unwrap();
    let new_logl = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(new_logl, ModelSearchCost::cost(&tkf_cost));
    assert_eq!(logl2, new_logl);
}

#[test]
fn tkf92_modify_subst_model_params_costs_match() {
    modify_tkf92_subst_params_costs_match_template::<K80>();
    modify_tkf92_subst_params_costs_match_template::<HKY>();
    modify_tkf92_subst_params_costs_match_template::<TN93>();
    modify_tkf92_subst_params_costs_match_template::<GTR>();
}

#[test]
fn tkf_modify_indel_model_params_costs_match() {
    modify_tkf92_indel_params_costs_match_template::<JC69>();
    modify_tkf92_indel_params_costs_match_template::<K80>();
    modify_tkf92_indel_params_costs_match_template::<HKY>();
    modify_tkf92_indel_params_costs_match_template::<TN93>();
    modify_tkf92_indel_params_costs_match_template::<GTR>();
    modify_tkf92_indel_params_costs_match_template::<WAG>();
    modify_tkf92_indel_params_costs_match_template::<BLOSUM>();
    modify_tkf92_indel_params_costs_match_template::<HIVB>();
}

#[test]
fn failing_test_to_see_github_action_output() {
    assert_eq!(1, 3);
}
