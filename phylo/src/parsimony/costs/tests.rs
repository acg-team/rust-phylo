use crate::alphabets::dna_alphabet;

use crate::parsimony::costs::parsimony_costs_model::ParsimonyCostsWModel;
use crate::parsimony::{
    costs::ParsimonyCostsSimple, BranchParsimonyCosts, GapMultipliers, ParsimonyCosts, Rounding,
};
use crate::substitution_models::{SubstModel, JC69, WAG};

#[test]
fn default_costs() {
    let def = ParsimonyCostsSimple::new_default();
    assert_eq!(def.branch.r#match(&b'A', &b'B'), 1.0);
    assert_eq!(def.branch.r#match(&b'B', &b'A'), 1.0);
    assert_eq!(def.branch.r#match(&b'N', &b'K'), 1.0);
    assert_eq!(def.branch.r#match(&b'C', &b'C'), 0.0);
    assert_eq!(def.branch.avg(), 1.0);
    assert_eq!(def.branch.gap_ext(), 0.5);
    assert_eq!(def.branch.gap_open(), 2.5);
}

#[test]
fn simple_costs() {
    let mismatch = 3.0;
    let gap_open = 2.0;
    let gap_ext = 10.5;
    let c = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext, dna_alphabet());
    assert_eq!(c.branch.r#match(&b'A', &b'B'), mismatch);
    assert_eq!(c.branch.r#match(&b'A', &b'A'), 0.0);
    assert_eq!(c.branch.avg(), mismatch);
    assert_eq!(c.branch.gap_open(), gap_open * mismatch);
    assert_eq!(c.branch.gap_ext(), gap_ext * mismatch);
    let mismatch = 1.0;
    let gap_open = 10.0;
    let gap_ext = 2.5;
    let c = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext, dna_alphabet());
    assert_eq!(c.branch.r#match(&b'A', &b'B'), mismatch);
    assert_eq!(c.branch.r#match(&b'A', &b'A'), 0.0);
    assert_eq!(c.branch.avg(), mismatch);
    assert_eq!(c.branch.gap_open(), gap_open);
    assert_eq!(c.branch.gap_ext(), gap_ext);
}

#[test]
fn protein_branch_scoring() {
    let gap_mult = GapMultipliers {
        open: 2.5,
        ext: 0.5,
    };
    let avg_01 = 5.7675;
    let avg_07 = 4.0075;
    let times = [0.1, 0.7];

    let model = SubstModel::<WAG>::new(&[], &[]);
    let cost =
        ParsimonyCostsWModel::new(model, &times, false, &gap_mult, &Rounding::zero()).unwrap();
    let branch_costs = cost.branch_costs(0.1);

    assert_eq!(branch_costs.avg(), avg_01);
    assert_eq!(branch_costs.gap_ext(), avg_01 * gap_mult.ext);
    assert_eq!(branch_costs.gap_open(), avg_01 * gap_mult.open);
    let branch_costs = cost.branch_costs(0.7);
    assert_eq!(branch_costs.avg(), avg_07);

    let model = SubstModel::<WAG>::new(&[], &[]);
    let cost =
        ParsimonyCostsWModel::new(model, &times, true, &gap_mult, &Rounding::zero()).unwrap();

    let branch_costs = cost.branch_costs(0.1);
    assert_eq!(branch_costs.avg(), avg_01);
    assert_eq!(branch_costs.cost_matrix().diagonal().sum(), 0.0);
    let branch_costs = cost.branch_costs(0.7);
    assert_ne!(branch_costs.avg(), avg_07);
    assert_eq!(branch_costs.cost_matrix().diagonal().sum(), 0.0);
}

#[test]
fn protein_scoring() {
    let gap_mult = GapMultipliers {
        open: 2.0,
        ext: 0.1,
    };

    let avg_01 = 5.7675;
    let avg_03 = 4.7475;
    let avg_05 = 4.2825;
    let avg_07 = 4.0075;
    let times = [0.1, 0.3, 0.5, 0.7];
    let model = SubstModel::<WAG>::new(&[], &[]);

    let cost =
        ParsimonyCostsWModel::new(model, &times, false, &gap_mult, &Rounding::zero()).unwrap();

    let branch_scores = cost.branch_costs(0.1);
    assert_eq!(branch_scores.avg(), avg_01);
    assert_eq!(branch_scores.gap_ext(), avg_01 * gap_mult.ext);
    assert_eq!(branch_scores.gap_open(), avg_01 * gap_mult.open);
    let branch_scores = cost.branch_costs(0.3);
    assert_eq!(branch_scores.avg(), avg_03);
    let branch_scores = cost.branch_costs(0.5);
    assert_eq!(branch_scores.avg(), avg_05);
    let branch_scores = cost.branch_costs(0.7);
    assert_eq!(branch_scores.avg(), avg_07);
}

#[test]
fn protein_branch_scoring_nearest() {
    let gap_mult = GapMultipliers {
        open: 2.0,
        ext: 0.1,
    };
    let avg_01 = 5.7675;
    let avg_05 = 4.2825;
    let times = [0.1, 0.5];
    let model = SubstModel::<WAG>::new(&[], &[]);

    let cost =
        ParsimonyCostsWModel::new(model, &times, false, &gap_mult, &Rounding::zero()).unwrap();

    let scores_01 = cost.branch_costs(0.1);
    assert_eq!(scores_01.avg(), avg_01);
    assert_eq!(scores_01.gap_ext(), avg_01 * gap_mult.ext);
    assert_eq!(scores_01.gap_open(), avg_01 * gap_mult.open);
    let scores_02 = cost.branch_costs(0.2);
    assert_eq!(scores_01.avg(), scores_02.avg());
    assert_eq!(scores_01.gap_ext(), scores_02.gap_ext());
    assert_eq!(scores_01.gap_open(), scores_02.gap_open());
    let scores_005 = cost.branch_costs(0.05);
    assert_eq!(scores_005.avg(), avg_01);
    let scores_015 = cost.branch_costs(0.15);
    assert_eq!(scores_015.avg(), avg_01);
    let scores_045 = cost.branch_costs(0.45);
    assert_eq!(scores_045.avg(), avg_05);
    let scores_05 = cost.branch_costs(0.5);
    assert_eq!(scores_05.avg(), avg_05);
    let scores_100 = cost.branch_costs(100.0);
    assert_eq!(scores_100.avg(), avg_05);
}

#[test]
fn dna_branch_scoring() {
    let gap_mult = GapMultipliers {
        open: 2.5,
        ext: 0.5,
    };
    let avg_01 = 2.25;
    let avg_07 = 1.75;
    let times = [0.1, 0.7];

    let model = SubstModel::<JC69>::new(&[], &[]);
    let cost =
        ParsimonyCostsWModel::new(model.clone(), &times, false, &gap_mult, &Rounding::zero())
            .unwrap();

    let b_cost = cost.branch_costs(0.1);

    assert_eq!(b_cost.avg(), avg_01);
    assert_eq!(b_cost.gap_ext(), avg_01 * gap_mult.ext);
    assert_eq!(b_cost.gap_open(), avg_01 * gap_mult.open);

    assert_eq!(cost.branch_costs(0.7).avg(), avg_07);

    let cost = ParsimonyCostsWModel::new(model.clone(), &times, true, &gap_mult, &Rounding::zero())
        .unwrap();

    let b_cost = cost.branch_costs(0.1);
    assert_eq!(b_cost.avg(), avg_01);
    assert_eq!(b_cost.cost_matrix().diagonal().sum(), 0.0);

    let b_cost = cost.branch_costs(0.7);
    println!("{:?}", b_cost.cost_matrix());

    assert_ne!(b_cost.avg(), avg_07);
    assert_eq!(b_cost.cost_matrix().diagonal().sum(), 0.0);
}

#[test]
fn dna_branch_scoring_nearest() {
    let gap_mult = GapMultipliers {
        open: 3.0,
        ext: 0.75,
    };
    let avg_01 = 2.25;
    let avg_07 = 1.75;
    let times = [0.1, 0.7];

    let model = SubstModel::<JC69>::new(&[], &[]);
    let cost =
        ParsimonyCostsWModel::new(model.clone(), &times, false, &gap_mult, &Rounding::zero())
            .unwrap();

    let scores_01 = cost.branch_costs(0.1);
    assert_eq!(scores_01.avg(), avg_01);
    assert_eq!(scores_01.gap_ext(), avg_01 * gap_mult.ext);
    assert_eq!(scores_01.gap_open(), avg_01 * gap_mult.open);
    let scores_08 = cost.branch_costs(0.8);
    assert_eq!(scores_08.avg(), avg_07);
    let scores_05 = cost.branch_costs(0.5);
    assert_eq!(scores_05.avg(), avg_07);
}
