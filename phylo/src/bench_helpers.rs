use std::collections::HashMap;

pub type BenchPath = &'static str;
pub type Paths = HashMap<&'static str, BenchPath>;

pub const DNA_EASY_5X1000: &str = "data/sim/GTR/gtr.fasta";
pub const DNA_EASY_8X1252: &str = "data/benchmark-datasets/dna/easy/wickd3b_7705.processed.fasta";
pub const DNA_EASY_17X2292: &str = "data/benchmark-datasets/dna/easy/wickd3a_7771.processed.fasta";
pub const DNA_EASY_33X4455: &str =
    "data/benchmark-datasets/dna/easy/tarvd7_ENSG00000126777.nuc.ord.processed.fasta";

pub const AA_EASY_6X97: &str = "data/benchmark-datasets/aa/easy/nagya1_Cluster9992.aln";
pub const AA_EASY_12X445: &str = "data/benchmark-datasets/aa/easy/wicka3_7004.aln";
pub const AA_EASY_14X165: &str = "data/benchmark-datasets/aa/easy/nagya1_Cluster5493.aln";
pub const AA_EASY_45X223: &str = "data/benchmark-datasets/aa/easy/whela7_Gene_0917.aln";
pub const AA_EASY_27X632: &str = "data/benchmark-datasets/aa/easy/boroa6_OG126_gene166.fasta";

pub const AA_MEDIUM_79X106: &str = "data/benchmark-datasets/aa/medium/strua5_gene1339_23221.aln";
pub const AA_MEDIUM_30X86: &str = "data/benchmark-datasets/aa/medium/nagya1_Cluster3439.aln";
