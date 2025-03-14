use crate::alphabets::dna_alphabet;

use crate::parsimony::costs::parsimony_costs_model::ParsimonyCostsWModel;
use crate::parsimony::{costs::ParsimonyCostsSimple, GapMultipliers, ParsimonyCosts, Rounding};
use crate::substitution_models::{SubstModel, JC69, WAG};

#[test]
fn default_costs() {
    let mismatch = 1.0;
    let gap = GapMultipliers {
        open: 2.5,
        ext: 0.5,
    };

    let c = ParsimonyCostsSimple::new(mismatch, gap.clone(), &dna_alphabet());
    assert_eq!(c.r#match(1.0, &b'A', &b'B'), mismatch);
    assert_eq!(c.r#match(1.0, &b'B', &b'A'), mismatch);
    assert_eq!(c.r#match(1.0, &b'N', &b'K'), mismatch);
    assert_eq!(c.r#match(1.0, &b'C', &b'C'), 0.0);
    assert_eq!(c.avg(2.0), mismatch);
    assert_eq!(c.gap_ext(3.0), gap.ext);
    assert_eq!(c.gap_open(2.0), gap.open);
}

#[test]
fn simple_costs() {
    let mismatch = 3.0;
    let gap = GapMultipliers {
        open: 2.0,
        ext: 10.5,
    };

    let c = ParsimonyCostsSimple::new(mismatch, gap.clone(), &dna_alphabet());
    assert_eq!(c.r#match(1.0, &b'A', &b'B'), mismatch);
    assert_eq!(c.r#match(3.0, &b'A', &b'A'), 0.0);
    assert_eq!(c.avg(4.9), mismatch);
    assert_eq!(c.gap_open(1.3), gap.open * mismatch);
    assert_eq!(c.gap_ext(2.4), gap.ext * mismatch);

    let mismatch = 1.0;

    let c = ParsimonyCostsSimple::new(mismatch, gap.clone(), &dna_alphabet());
    assert_eq!(c.r#match(1.0, &b'A', &b'B'), mismatch);
    assert_eq!(c.r#match(1.0, &b'A', &b'A'), 0.0);
    assert_eq!(c.avg(1.0), mismatch);
    assert_eq!(c.gap_open(1.0), gap.open);
    assert_eq!(c.gap_ext(1.0), gap.ext);
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
        ParsimonyCostsWModel::new(&model, &times, false, &gap_mult, &Rounding::zero()).unwrap();

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.gap_ext(0.1), avg_01 * gap_mult.ext);
    assert_eq!(cost.gap_open(0.1), avg_01 * gap_mult.open);

    assert_eq!(cost.avg(0.7), avg_07);

    let model = SubstModel::<WAG>::new(&[], &[]);
    let cost =
        ParsimonyCostsWModel::new(&model, &times, true, &gap_mult, &Rounding::zero()).unwrap();

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.r#match(0.1, &b'A', &b'A'), 0.0);

    assert_ne!(cost.avg(0.7), avg_07);
    assert_eq!(cost.r#match(0.7, &b'A', &b'A'), 0.0);
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
        ParsimonyCostsWModel::new(&model, &times, false, &gap_mult, &Rounding::zero()).unwrap();

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.gap_ext(0.1), avg_01 * gap_mult.ext);
    assert_eq!(cost.gap_open(0.1), avg_01 * gap_mult.open);
    assert_eq!(cost.avg(0.3), avg_03);
    assert_eq!(cost.avg(0.5), avg_05);
    assert_eq!(cost.avg(0.7), avg_07);
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
        ParsimonyCostsWModel::new(&model, &times, false, &gap_mult, &Rounding::zero()).unwrap();

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.gap_ext(0.1), avg_01 * gap_mult.ext);
    assert_eq!(cost.gap_open(0.1), avg_01 * gap_mult.open);
    assert_eq!(cost.avg(0.1), cost.avg(0.2));
    assert_eq!(cost.gap_ext(0.1), cost.gap_ext(0.2));
    assert_eq!(cost.gap_open(0.1), cost.gap_open(0.2));
    assert_eq!(cost.avg(0.005), avg_01);
    assert_eq!(cost.avg(0.15), avg_01);
    assert_eq!(cost.avg(0.45), avg_05);
    assert_eq!(cost.avg(0.5), avg_05);
    assert_eq!(cost.avg(100.0), avg_05);
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
        ParsimonyCostsWModel::new(&model, &times, false, &gap_mult, &Rounding::zero()).unwrap();

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.gap_ext(0.1), avg_01 * gap_mult.ext);
    assert_eq!(cost.gap_open(0.1), avg_01 * gap_mult.open);

    assert_eq!(cost.avg(0.7), avg_07);

    let cost =
        ParsimonyCostsWModel::new(&model, &times, true, &gap_mult, &Rounding::zero()).unwrap();

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.r#match(0.1, &b'N', &b'N'), 0.0);

    assert_ne!(cost.avg(0.7), avg_07);
    assert_eq!(cost.r#match(0.7, &b'N', &b'N'), 0.0);
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
        ParsimonyCostsWModel::new(&model, &times, false, &gap_mult, &Rounding::zero()).unwrap();

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.gap_ext(0.1), avg_01 * gap_mult.ext);
    assert_eq!(cost.gap_open(0.1), avg_01 * gap_mult.open);

    assert_eq!(cost.avg(0.8), avg_07);

    assert_eq!(cost.avg(0.5), avg_07);
}
