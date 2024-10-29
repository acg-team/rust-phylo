use rstest::*;

use rand::Rng;

use crate::evolutionary_models::{
    DNAModelType::{self, UNDEF as DNAUNDEF, *},
    ProteinModelType::{self, UNDEF as ProteinUNDEF, *},
};

#[cfg(test)]
fn random_capitalise(input: &str) -> String {
    let mut rng = rand::thread_rng();
    input
        .chars()
        .map(|c| {
            if c.is_alphabetic() && rng.gen_bool(0.5) {
                c.to_uppercase().collect::<String>()
            } else {
                c.to_string()
            }
        })
        .collect()
}

#[rstest]
#[case::jc69("jc69", JC69, &["jc70"])]
#[case::k80("k80", K80, &["K 80", "k89"])]
#[case::hky("hky", HKY, &["hkz", "hky3"])]
#[case::tn93("tn93", TN93, &["TN92", "tn993"])]
#[case::gtr("gtr", GTR, &["ctr", "GTP", "gtr1"])]
fn dna_type_by_name(
    #[case] name: &str,
    #[case] model_type: DNAModelType,
    #[case] wrong_names: &[&str],
) {
    assert_eq!(DNAModelType::from(name), model_type);
    assert_eq!(
        DNAModelType::from(name.to_ascii_uppercase().as_str()),
        model_type
    );
    for _ in 0..10 {
        assert_eq!(
            DNAModelType::from(random_capitalise(name).as_str()),
            model_type
        );
    }
    for &name in wrong_names {
        assert_eq!(DNAModelType::from(name), DNAUNDEF);
    }
}

#[rstest]
#[case::wag("wag", WAG, &["weg", "wag1"])]
#[case::blosum("blosum", BLOSUM, &["BLOS", "blosum1", "blosum62"])]
#[case::hivb("hivb", HIVB, &["Hiv1", "HIB", "HIVA"])]
fn protein_type_by_name(
    #[case] name: &str,
    #[case] model_type: ProteinModelType,
    #[case] wrong_names: &[&str],
) {
    assert_eq!(ProteinModelType::from(name), model_type);
    assert_eq!(
        ProteinModelType::from(name.to_ascii_uppercase().as_str()),
        model_type
    );
    for _ in 0..10 {
        assert_eq!(
            ProteinModelType::from(random_capitalise(name).as_str()),
            model_type
        );
    }
    for &name in wrong_names {
        assert_eq!(ProteinModelType::from(name), ProteinUNDEF);
    }
}

#[test]
fn dna_type_by_name_given_protein() {
    assert_eq!(DNAModelType::from("wag"), DNAUNDEF);
    assert_eq!(DNAModelType::from("BLOSUM"), DNAUNDEF);
    assert_eq!(DNAModelType::from("HIv"), DNAUNDEF);
}

#[test]
fn protein_type_by_name_given_dna() {
    assert_eq!(ProteinModelType::from("k80"), ProteinUNDEF);
    assert_eq!(ProteinModelType::from("gtr"), ProteinUNDEF);
    assert_eq!(ProteinModelType::from("TN93"), ProteinUNDEF);
    assert_eq!(ProteinModelType::from("waq"), ProteinUNDEF);
    assert_eq!(ProteinModelType::from("HIV"), ProteinUNDEF);
}
