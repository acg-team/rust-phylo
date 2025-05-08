use crate::alphabets::{AMINOACIDS, NUCLEOTIDES};
use crate::parsimony::{
    DiagonalZeros as Z, GapCost, ModelScoringBuilder as MCB, ParsimonyScoring, Rounding as R,
    SimpleScoring,
};
use crate::substitution_models::{SubstModel, JC69, WAG};

#[test]
fn default_costs() {
    let mismatch = 1.0;
    let gap = GapCost {
        open: 2.5,
        ext: 0.5,
    };

    let c = SimpleScoring::new(mismatch, gap.clone());
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
    let gap = GapCost {
        open: 2.0,
        ext: 10.5,
    };

    let c = SimpleScoring::new(mismatch, gap.clone());
    for char in AMINOACIDS.iter() {
        for char2 in AMINOACIDS.iter() {
            if char == char2 {
                assert_eq!(c.r#match(1.0, char, char2), 0.0);
            } else {
                assert_eq!(c.r#match(1.0, char, char2), mismatch);
            }
        }
    }
    assert_eq!(c.avg(4.9), mismatch);
    assert_eq!(c.gap_open(1.3), gap.open * mismatch);
    assert_eq!(c.gap_ext(2.4), gap.ext * mismatch);

    let mismatch = 1.0;

    let c = SimpleScoring::new(mismatch, gap.clone());
    for char in NUCLEOTIDES.iter() {
        for char2 in NUCLEOTIDES.iter() {
            if char == char2 {
                assert_eq!(c.r#match(1.0, char, char2), 0.0);
            } else {
                assert_eq!(c.r#match(1.0, char, char2), mismatch);
            }
        }
    }
    assert_eq!(c.avg(1.0), mismatch);
    assert_eq!(c.gap_open(1.0), gap.open);
    assert_eq!(c.gap_ext(1.0), gap.ext);
}

#[test]
fn protein_branch_scoring() {
    let gap = GapCost {
        open: 2.5,
        ext: 0.5,
    };

    let times = vec![0.1, 0.7];

    let model = SubstModel::<WAG>::new(&[], &[]);
    let cost = MCB::new(model.clone())
        .gap_cost(gap.clone())
        .rounding(R::zero())
        .times(times.clone())
        .build()
        .unwrap();

    let avg_01 = 5.7675;
    let avg_07 = 4.0075;

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.gap_ext(0.1), avg_01 * gap.ext);
    assert_eq!(cost.gap_open(0.1), avg_01 * gap.open);

    assert_eq!(cost.avg(0.7), avg_07);

    let cost = MCB::new(model)
        .gap_cost(gap)
        .diagonal(Z::zero())
        .rounding(R::zero())
        .times(times)
        .build()
        .unwrap();

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.r#match(0.1, &b'A', &b'A'), 0.0);

    assert_ne!(cost.avg(0.7), avg_07);
    assert_eq!(cost.r#match(0.7, &b'A', &b'A'), 0.0);
}

#[test]
fn protein_scoring() {
    let gap = GapCost {
        open: 2.0,
        ext: 0.1,
    };

    let times = vec![0.1, 0.3, 0.5, 0.7];
    let model = SubstModel::<WAG>::new(&[], &[]);
    let cost = MCB::new(model)
        .gap_cost(gap.clone())
        .rounding(R::zero())
        .times(times)
        .build()
        .unwrap();

    let avg_01 = 5.7675;
    let avg_03 = 4.7475;
    let avg_05 = 4.2825;
    let avg_07 = 4.0075;

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.gap_ext(0.1), avg_01 * gap.ext);
    assert_eq!(cost.gap_open(0.1), avg_01 * gap.open);
    assert_eq!(cost.avg(0.3), avg_03);
    assert_eq!(cost.avg(0.5), avg_05);
    assert_eq!(cost.avg(0.7), avg_07);
}

#[test]
fn protein_branch_scoring_nearest() {
    let gap = GapCost {
        open: 2.0,
        ext: 0.1,
    };
    let times = vec![0.1, 0.5];
    let model = SubstModel::<WAG>::new(&[], &[]);

    let cost = MCB::new(model)
        .gap_cost(gap.clone())
        .times(times)
        .rounding(R::zero())
        .build()
        .unwrap();

    let avg_01 = 5.7675;
    let avg_05 = 4.2825;

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.gap_ext(0.1), avg_01 * gap.ext);
    assert_eq!(cost.gap_open(0.1), avg_01 * gap.open);
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
    let gap = GapCost {
        open: 2.5,
        ext: 0.5,
    };
    let times = vec![0.1, 0.7];

    let model = SubstModel::<JC69>::new(&[], &[]);
    let cost = MCB::new(model.clone())
        .gap_cost(gap.clone())
        .times(times.clone())
        .rounding(R::zero())
        .build()
        .unwrap();

    let avg_01 = 2.25;
    let avg_07 = 1.75;

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.gap_ext(0.1), avg_01 * gap.ext);
    assert_eq!(cost.gap_open(0.1), avg_01 * gap.open);

    assert_eq!(cost.avg(0.7), avg_07);

    let cost = MCB::new(model.clone())
        .gap_cost(gap.clone())
        .times(times)
        .diagonal(Z::zero())
        .rounding(R::zero())
        .build()
        .unwrap();

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.r#match(0.1, &b'N', &b'N'), 0.0);

    assert_ne!(cost.avg(0.7), avg_07);
    assert_eq!(cost.r#match(0.7, &b'N', &b'N'), 0.0);
}

#[test]
fn dna_branch_scoring_nearest() {
    let gap = GapCost {
        open: 3.0,
        ext: 0.75,
    };
    let times = vec![0.1, 0.7];

    let model = SubstModel::<JC69>::new(&[], &[]);
    let cost = MCB::new(model)
        .gap_cost(gap.clone())
        .times(times)
        .rounding(R::zero())
        .build()
        .unwrap();

    let avg_01 = 2.25;
    let avg_07 = 1.75;

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.gap_ext(0.1), avg_01 * gap.ext);
    assert_eq!(cost.gap_open(0.1), avg_01 * gap.open);

    assert_eq!(cost.avg(0.8), avg_07);

    assert_eq!(cost.avg(0.5), avg_07);
}
