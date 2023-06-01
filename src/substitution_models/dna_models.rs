use crate::sequences::{charify, NUCLEOTIDES_STR};
use crate::substitution_models::{FreqVector, SubstMatrix};

type DNASubstMatrix = SubstMatrix<4>;
type DNAFreqVector = FreqVector<4>;

pub(crate) fn nucleotide_index() -> [i32; 255] {
    let mut index = [-1 as i32; 255];
    for (i, char) in charify(NUCLEOTIDES_STR).into_iter().enumerate() {
        index[char as usize] = i as i32;
        index[char.to_ascii_lowercase() as usize] = i as i32;
    }
    index
}

pub(crate) fn jc69() -> (DNASubstMatrix, DNAFreqVector) {
    (
        DNASubstMatrix::from(JC69_ARR),
        DNAFreqVector::from(JC69_PI_ARR),
    )
}

pub(crate) fn k80(alpha: f64, beta: f64) -> (DNASubstMatrix, DNAFreqVector) {
    (
        DNASubstMatrix::from([
            [
                -1.0,
                alpha / (alpha + 2.0 * beta),
                beta / (alpha + 2.0 * beta),
                beta / (alpha + 2.0 * beta),
            ],
            [
                alpha / (alpha + 2.0 * beta),
                -1.0,
                beta / (alpha + 2.0 * beta),
                beta / (alpha + 2.0 * beta),
            ],
            [
                beta / (alpha + 2.0 * beta),
                beta / (alpha + 2.0 * beta),
                -1.0,
                alpha / (alpha + 2.0 * beta),
            ],
            [
                beta / (alpha + 2.0 * beta),
                beta / (alpha + 2.0 * beta),
                alpha / (alpha + 2.0 * beta),
                -1.0,
            ],
        ]),
        DNAFreqVector::from(JC69_PI_ARR),
    )
}

pub(crate) fn gtr(
    f_t: f64,
    f_c: f64,
    f_a: f64,
    f_g: f64,
    r_tc: f64,
    r_ta: f64,
    r_tg: f64,
    r_ca: f64,
    r_cg: f64,
    r_ag: f64,
) -> (DNASubstMatrix, DNAFreqVector) {
    assert_f64_near!(f_t + f_c + f_a + f_g, 1.0);
    (
        DNASubstMatrix::from([
            [
                -(r_tc * f_c + r_ta * f_a + r_tg * f_g),
                r_tc * f_c,
                r_ta * f_a,
                r_tg * f_g,
            ],
            [
                r_tc * f_t,
                -(r_tc * f_t + r_ca * f_a + r_cg * f_g),
                r_ca * f_a,
                r_cg * f_g,
            ],
            [
                r_ta * f_t,
                r_ca * f_c,
                -(r_ta * f_t + r_ca * f_c + r_ag * f_g),
                r_ag * f_g,
            ],
            [
                r_tg * f_t,
                r_cg * f_c,
                r_ag * f_a,
                -(r_tg * f_t + r_cg * f_c + r_ag * f_a),
            ],
        ]),
        DNAFreqVector::from([f_t, f_c, f_a, f_g]),
    )
}

const JC69_ARR: [[f64; 4]; 4] = [
    [-1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    [1.0 / 3.0, -1.0, 1.0 / 3.0, 1.0 / 3.0],
    [1.0 / 3.0, 1.0 / 3.0, -1.0, 1.0 / 3.0],
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, -1.0],
];

const JC69_PI_ARR: [f64; 4] = [0.25, 0.25, 0.25, 0.25];
