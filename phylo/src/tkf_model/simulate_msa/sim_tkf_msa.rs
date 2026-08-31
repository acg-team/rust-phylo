use bio::io::fasta::Record;
use rand::{Rng, RngCore, SeedableRng};

use crate::alignment::{Alignment, AlignmentSimulation, AncestralAlignment, Sequences, MASA};
use crate::alphabets::GAP;
use crate::random::RandomGenerator;
use crate::substitution_models::{QMatrix, SubstModel, SubstitutionSimulator};
use crate::tkf_model::simulate_msa::sim_tkf_indel_msa::{
    FragmentSampler, TKFIndelMSASimulationResult, TKFIndelMSASimulator,
};
use crate::tkf_model::simulate_msa::{ExpectedRootLength, Fragmentation, RootLength};
use crate::tkf_model::TKFModel;
use crate::tree::{NodeIdx::Internal, NodeIdx::Leaf, Tree};
use crate::{record_wo_desc as record, Result};

/// Simulates a full TKF process: first indels then substitutions.
///
/// The simulator runs the [indel simulator](TKFIndelMSASimulator) to obtain an ancestral alignment
/// that represents homology paths of character presence, then simulates substitutions along the same
/// tree for the number of columns produced by the indel simulation and finally uses the indel MSA
/// as a mask to place gaps.
/// Note, that the MASA might contain columns where the character goes extinct, i.e., all leaf
/// sequences have a gap in that column. You may want to call
/// [`AncestralAlignment::remove_extinct_columns`](`crate::alignment::AncestralAlignment::remove_extinct_columns`) on the
/// resulting alignment or
/// [`TKFMSASimulationResult::remove_extinct_columns`](`TKFMSASimulationResult::remove_extinct_columns`) on the
/// simulation result if you also care about the fragmentation and want to remove those.
pub struct TKFMSASimulator<T, R>
where
    T: TKFModel + FragmentSampler + ExpectedRootLength,
    R: Rng + SeedableRng + RngCore,
{
    indel_sim: TKFIndelMSASimulator<T, R>,
    subst_sim: SubstitutionSimulator<R>,
}

impl<T, R> TKFMSASimulator<T, R>
where
    T: TKFModel + FragmentSampler + ExpectedRootLength,
    R: Rng + SeedableRng + RngCore + Clone,
{
    /// Create a new TKFMSASimulator with the given indel model, substitution model, tree, RNG and
    /// max insertion length (i.e., the max number of inserted links (=fragments) in a single event;
    /// since fragments can consist of multiple characters the number of inserted characters can be
    /// longer than this max length).
    pub fn new<Q: QMatrix>(
        indel_model: T,
        subst_model: SubstModel<Q>,
        tree: Tree,
        rng: RandomGenerator<R>,
        max_insertion_length: usize,
    ) -> Self {
        let indel_sim =
            TKFIndelMSASimulator::new(indel_model, tree.clone(), rng.clone(), max_insertion_length);
        let dummy_len = 1;
        let subst_sim = SubstitutionSimulator::new(subst_model, tree, rng, dummy_len).unwrap();
        Self {
            indel_sim,
            subst_sim,
        }
    }

    /// Sets a defined root length for the simulation. If `None`, the root length is sampled.
    pub fn root_length(&mut self, root_length: RootLength) -> &mut Self {
        self.indel_sim.root_length(root_length);
        self
    }

    /// Simulates the full TKF process (indels then substitutions) and returns both
    /// the MSA and the fragmentation (right-exclusive boundaries of fragments).
    pub fn simulate_with_fragments<AA: AncestralAlignment>(&self) -> TKFMSASimulationResult<AA> {
        // First, indel simulation
        let TKFIndelMSASimulationResult {
            masa: indel_msa,
            fragmentation,
        } = self.indel_sim.simulate_with_fragments::<AA>();

        // Second, substitution simulation
        let aln_len = indel_msa.len();
        let subst_msa: AA = self
            .subst_sim
            .simulate_ancestral_alignment_with_length(aln_len);

        // Third, mask the substitution msa with gaps from the indel msa
        // Construct a sequences vector including ancestral and leaf records
        let mut combined_records: Vec<Record> = Vec::new();
        for node in self.indel_sim.tree().preorder() {
            let id = self.indel_sim.tree().node(node).id.clone();
            // get mask_seq (from indel msa) and subst_seq (from substitution msa)
            let mask_mapping = match node {
                Internal(_) => indel_msa.ancestral_map(node),
                Leaf(_) => indel_msa.leaf_map(node),
            };
            let subst_seq = match node {
                Leaf(_) => subst_msa.seqs().record_by_id(&id).seq(),
                Internal(_) => subst_msa.ancestral_seqs().record_by_id(&id).seq(),
            };
            debug_assert!(
                mask_mapping.len() == subst_seq.len(),
                "Mask and substitution sequences must be the same length"
            );
            // apply mask: if mask is a gap, put a gap; otherwise keep the subst character
            let final_seq: Vec<u8> = mask_mapping
                .iter()
                .zip(subst_seq.iter())
                .map(
                    |(mask, subst_char)| {
                        if mask.is_some() {
                            *subst_char
                        } else {
                            GAP
                        }
                    },
                )
                .collect();

            combined_records.push(record!(&id, &final_seq));
        }

        // Lastly, construct the final ancestral MSA from the combined records
        let seqs = Sequences::new(combined_records);
        let masa = AA::from_aligned_with_ancestral(seqs, self.indel_sim.tree()).unwrap();
        TKFMSASimulationResult {
            masa,
            fragmentation,
        }
    }
}

/// Result of a full TKF simulation (indels + substitutions) including fragmentation.
pub struct TKFMSASimulationResult<AA: AncestralAlignment> {
    pub masa: AA,
    pub fragmentation: Fragmentation,
}

impl<AA: AncestralAlignment> TKFMSASimulationResult<AA> {
    pub fn remove_extinct_columns(&mut self) -> Result<()> {
        let keep_col_mask = self.masa.remove_extinct_columns();
        self.fragmentation.remove_cols(&keep_col_mask)
    }
}

impl<T, R> AlignmentSimulation for TKFMSASimulator<T, R>
where
    T: TKFModel + FragmentSampler + ExpectedRootLength,
    R: Rng + SeedableRng + RngCore + Clone,
{
    fn simulate_ancestral_alignment<AA: AncestralAlignment>(&self) -> AA {
        self.simulate_with_fragments::<AA>().masa
    }

    fn simulate_alignment<A: Alignment>(&self) -> A {
        self.simulate_ancestral_alignment::<MASA>()
            .into_alignment(self.indel_sim.tree())
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use hashbrown::HashSet;

    use crate::alignment::{Alignment, AncestralAlignment, MASA};
    use crate::phylo_info::PhyloInfo;
    use crate::random::DefaultGenerator;
    use crate::substitution_models::{dna_models::GTR, SubstModel};
    use crate::tkf_model::{TKF91IndelModel, TKF92IndelModel};
    use crate::tree;

    use super::*;

    /// In the [`tkf92_simulation`] test, only A <-> T and G <-> C transitions are allowed.
    /// Checks that the simulation respects the mutation constraints of the GTR model used.
    /// So each column must contain at most two unique characters, which must be a valid pair.
    #[cfg(test)]
    fn check_mutation_constraints(msa: &MASA, tree: &Tree) {
        for col_idx in 0..msa.len() {
            let mut col_chars = HashSet::new();
            for node_idx in tree.preorder() {
                let node_id = tree.node_id(node_idx);
                let seq = match node_idx {
                    Leaf(_) => msa.seqs().record_by_id(node_id).seq(),
                    Internal(_) => msa.ancestral_seqs().record_by_id(node_id).seq(),
                };
                let map = match node_idx {
                    Internal(_) => msa.ancestral_map(node_idx),
                    Leaf(_) => msa.leaf_map(node_idx),
                };

                if let Some(pos) = map[col_idx] {
                    col_chars.insert(seq[pos]);
                }
            }
            assert!(
                col_chars.len() <= 2,
                "Column {} has too many unique characters: {:?}",
                col_idx,
                col_chars
            );
            // Verify specific pairings if multiple characters exist
            if col_chars.len() == 2 {
                let chars: Vec<u8> = col_chars.into_iter().collect();
                let c1 = chars[0];
                let c2 = chars[1];
                let valid_pair = matches!(
                    (c1, c2),
                    (b'A', b'T') | (b'T', b'A') | (b'C', b'G') | (b'G', b'C')
                );
                assert!(
                    valid_pair,
                    "Invalid mutation pair in column {}: {} and {}",
                    col_idx, c1 as char, c2 as char
                );
            }
        }
    }

    #[test]
    fn tkf92_simulation() {
        let tree = tree!(
            "((((A:1.0,B:1.0)I1:0.5,C:1.5)I2:0.5,D:2.0)I3:0.5,((E:1.0,F:1.0)I4:0.5,G:1.5)I5:0.5)R;"
        );

        let freqs = [0.25, 0.25, 0.25, 0.25];
        let params = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let subst_model = SubstModel::<GTR>::new(&freqs, &params);

        let lambda = 0.19;
        let mu = 0.2;
        let r = 0.8;
        let tkf_model = TKF92IndelModel::new(lambda, mu, r);

        let max_insertion_length = 50;
        let simulator = TKFMSASimulator::new(
            tkf_model,
            subst_model,
            tree.clone(),
            DefaultGenerator::new(123),
            max_insertion_length,
        );

        let msa = simulator.simulate_ancestral_alignment::<MASA>();

        assert_eq!(msa.seq_count() + msa.ancestral_seqs().len(), 13);
        assert_eq!(msa.seq_count(), 7); // A, B, C, D, E, F, G
        assert!(msa.len() > 1);

        check_mutation_constraints(&msa, &tree);

        let phylo = PhyloInfo {
            msa: msa.clone(),
            tree,
        };
        assert!(
            phylo.check_dollos_constraint().is_ok(),
            "Simulated alignment must satisfy Dollo's constraint (no re-gain of characters)"
        );
    }

    #[test]
    fn tkf91_simulation_fixed_root_length() {
        let tree = tree!("(A:1.0,B:1.0)R;");
        let subst_model = SubstModel::<GTR>::new(&[0.25; 4], &[1.0; 6]);
        let tkf_model = TKF91IndelModel::new(0.1, 0.2);

        let mut simulator = TKFMSASimulator::new(
            tkf_model,
            subst_model,
            tree.clone(),
            DefaultGenerator::new(123),
            50,
        );
        simulator.root_length(RootLength::Defined(100));

        let msa = simulator.simulate_ancestral_alignment::<MASA>();
        let root_map = msa.ancestral_map(&tree.root);
        assert_eq!(root_map.iter().filter(|s| s.is_some()).count(), 100);
    }

    #[test]
    fn tkf92_simulation_fixed_root_length() {
        let tree = tree!("(A:1.0,B:1.0)R;");
        let subst_model = SubstModel::<GTR>::new(&[0.25; 4], &[1.0; 6]);
        let tkf_model = TKF92IndelModel::new(0.1, 0.2, 0.8);

        let mut simulator = TKFMSASimulator::new(
            tkf_model,
            subst_model,
            tree.clone(),
            DefaultGenerator::new(123),
            50,
        );
        simulator.root_length(RootLength::Defined(100));

        let msa = simulator.simulate_ancestral_alignment::<MASA>();
        let root_map = msa.ancestral_map(&tree.root);
        assert_eq!(root_map.iter().filter(|s| s.is_some()).count(), 100);
    }
}
