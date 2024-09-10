use std::path::{Path, PathBuf};

use approx::assert_relative_eq;
use bio::io::fasta::Record;

use crate::alignment::Sequences;
use crate::evolutionary_models::{DNAModelType::*, EvoModel, ProteinModelType::*};
use crate::likelihood::PhyloCostFunction;
use crate::optimisers::{BranchOptimiser, PhyloOptimiser, TopologyOptimiser};
use crate::phylo_info::PhyloInfoBuilder;
use crate::pip_model::PIPModel;
use crate::substitution_models::{DNASubstModel, ProteinSubstModel};
use crate::tree::tree_parser::from_newick_string;

#[test]
fn k80_topo_optimisation() {
    let tree = from_newick_string(&String::from(
        "(((A:1.0,B:1.0)E:2.0,(C:1.0,D:1.0)F:2.0)G:3.0);",
    ))
    .unwrap()[0]
        .clone();
    let sequences = Sequences::new(vec![
        Record::with_attrs("A", None, b"CTATATATAC"),
        Record::with_attrs("B", None, b"ATATATATAA"),
        Record::with_attrs("C", None, b"TTATATATAT"),
        Record::with_attrs("D", None, b"TTATATATAT"),
    ]);
    let info = &PhyloInfoBuilder::build_from_objects(sequences, tree).unwrap();
    let k80 = DNASubstModel::new(K80, &[4.0, 1.0]).unwrap();
    let unopt_logl = k80.cost(info);
    let o = TopologyOptimiser::new(&k80, info).run().unwrap();

    k80.reset();
    assert!(o.final_logl >= unopt_logl);
    assert_relative_eq!(o.final_logl, k80.cost(&o.i));
}

#[test]
fn k80_sim_topo_optimisation_from_nj() {
    let info = &PhyloInfoBuilder::new(PathBuf::from("./data/sim/K80/K80.fasta"))
        .build()
        .unwrap();
    let k80 = DNASubstModel::new(K80, &[4.0, 1.0]).unwrap();
    let unopt_logl = k80.cost(info);
    let o = TopologyOptimiser::new(&k80, info).run().unwrap();

    k80.reset();
    assert!(o.final_logl >= unopt_logl);
    assert_relative_eq!(o.final_logl, k80.cost(&o.i));
    assert_relative_eq!(o.final_logl, -4060.91963, epsilon = 1e-5);
}

#[test]
fn topology_opt_simulated_from_tree() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/K80/K80.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let model = DNASubstModel::new(JC69, &[]).unwrap();
    let unopt_logl = model.cost(&info);
    let o = TopologyOptimiser::new(&model, &info).run().unwrap();

    model.reset();
    assert!(o.final_logl >= unopt_logl);
    assert_relative_eq!(o.final_logl, model.cost(&o.i), epsilon = 1e-5);

    let phyml_info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/K80/K80.fasta"),
        PathBuf::from("./data/sim/K80/phyml_tree.newick"),
    )
    .build()
    .unwrap();

    model.reset();
    assert_relative_eq!(o.final_logl, model.cost(&phyml_info), epsilon = 1e-5);
    assert_relative_eq!(o.final_logl, -4038.721121221992, epsilon = 1e-5);

    let taxa = ["Gorilla", "Orangutan", "Gibbon", "Human", "Chimpanzee"];
    for taxon in taxa.iter() {
        assert_relative_eq!(
            o.i.tree.node(&o.i.tree.idx(taxon)).blen,
            phyml_info.tree.node(&phyml_info.tree.idx(taxon)).blen,
            epsilon = 1e-5
        );
    }
    assert_relative_eq!(o.i.tree.height, phyml_info.tree.height, epsilon = 1e-5);
    assert_eq!(o.i.tree.robinson_foulds(&phyml_info.tree), 0);
}

#[test]
fn topology_opt_simulated_from_wrong_tree() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/K80/K80.fasta"),
        PathBuf::from("./data/sim/wrong_tree.newick"),
    )
    .build()
    .unwrap();
    let model = DNASubstModel::new(JC69, &[]).unwrap();
    let unopt_logl = model.cost(&info);
    let o = TopologyOptimiser::new(&model, &info).run().unwrap();

    model.reset();
    assert!(o.final_logl >= unopt_logl);
    assert_relative_eq!(o.final_logl, model.cost(&o.i), epsilon = 1e-5);

    let phyml_info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/K80/K80.fasta"),
        PathBuf::from("./data/sim/K80/phyml_tree.newick"),
    )
    .build()
    .unwrap();
    model.reset();
    assert_relative_eq!(o.final_logl, model.cost(&phyml_info), epsilon = 1e-5);
    assert_relative_eq!(o.final_logl, -4038.721121221992, epsilon = 1e-5);

    let taxa = ["Gorilla", "Orangutan", "Gibbon", "Chimpanzee"];
    for taxon in taxa.iter() {
        assert_relative_eq!(
            o.i.tree.by_id(taxon).blen,
            phyml_info.tree.by_id(taxon).blen,
            epsilon = 1e-5
        );
    }
    // Slightly different rooting leads to a different breakdown of branches,
    // but sum is still same
    assert_relative_eq!(
        o.i.tree.by_id("Human").blen
            + o.i
                .tree
                .node(&(o.i.tree.by_id("Chimpanzee")).parent.unwrap())
                .blen,
        phyml_info.tree.by_id("Human").blen,
        epsilon = 1e-5
    );
    assert_relative_eq!(o.i.tree.height, phyml_info.tree.height, epsilon = 1e-4);
    assert_eq!(o.i.tree.robinson_foulds(&phyml_info.tree), 0);
}

#[test]
fn protein_topology_optimisation_given_tree_start() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let out_tree = fldr.join("optimisation_tree_start.newick");
    let phyml_info = PhyloInfoBuilder::with_attrs(
        fldr.join("nogap_seqs.fasta"),
        fldr.join("phyml_result.newick"),
    )
    .build()
    .unwrap();
    let wag = ProteinSubstModel::new(WAG, &[]).unwrap();
    let phyml_logl = wag.cost(&phyml_info);

    // use crate::io::write_newick_to_file;
    // let info = PhyloInfoBuilder::with_attrs(fldr.join("nogap_seqs.fasta"), fldr.join("tree.newick"))
    //     .build()
    //     .unwrap();
    // let wag = ProteinSubstModel::new(WAG, &[]).unwrap();
    // let unopt_logl = wag.cost(&info);
    // let o = TopologyOptimiser::new(&wag, &info).run().unwrap();
    // let _ = write_newick_to_file(&[o.i.tree.clone()], out_tree);
    // assert!(o.final_logl >= unopt_logl);
    // let tree = o.i.tree;
    // let logl = o.final_logl;
    // Ran in ~215.21s
    // PhyML runs in 3s

    // Optimisation can take long, tree checked is output of above
    let info = PhyloInfoBuilder::with_attrs(fldr.join("nogap_seqs.fasta"), out_tree)
        .build()
        .unwrap();
    wag.reset();
    let logl = wag.cost(&info);
    let tree = info.tree;

    // Compare tree and logl to PhyML output
    assert_relative_eq!(tree.height, phyml_info.tree.height, epsilon = 1e-3);
    assert_eq!(tree.robinson_foulds(&phyml_info.tree), 0);
    assert_relative_eq!(logl, phyml_logl, epsilon = 1e0);
}

#[test]
fn protein_topology_optimisation_nj_start() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let out_tree = fldr.join("optimisation_nj_start.newick");
    let phyml_info = PhyloInfoBuilder::with_attrs(
        fldr.join("nogap_seqs.fasta"),
        fldr.join("phyml_result.newick"),
    )
    .build()
    .unwrap();
    let wag = ProteinSubstModel::new(WAG, &[]).unwrap();
    let phyml_logl = wag.cost(&phyml_info);

    // use crate::io::write_newick_to_file;
    // let info = PhyloInfoBuilder::new(fldr.join("nogap_seqs.fasta"))
    //     .build()
    //     .unwrap();
    // let wag = ProteinSubstModel::new(WAG, &[]).unwrap();
    // let unopt_logl = wag.cost(&info);
    // let o = TopologyOptimiser::new(&wag, &info).run().unwrap();
    // let _ = write_newick_to_file(&[o.i.tree.clone()], out_tree);
    // assert!(o.final_logl >= unopt_logl);
    // let tree = o.i.tree;
    // let logl = o.final_logl;
    // Ran in ~218.05s

    // Optimisation itself can take long, tree checked is output of above
    let info = PhyloInfoBuilder::with_attrs(fldr.join("nogap_seqs.fasta"), out_tree)
        .build()
        .unwrap();
    wag.reset();
    let logl = wag.cost(&info);
    let tree = info.tree;

    // Compare tree height and logl to the output of PhyML
    assert_relative_eq!(tree.height, phyml_info.tree.height, epsilon = 1e-3);
    assert_eq!(tree.robinson_foulds(&phyml_info.tree), 0);
    assert_relative_eq!(logl, phyml_logl, epsilon = 1e0);
}

#[test]
fn pip_vs_subst_dna_tree() {
    let info = &PhyloInfoBuilder::new(PathBuf::from("./data/sim/K80/K80.fasta"))
        .build()
        .unwrap();
    let pip = PIPModel::<DNASubstModel>::new(K80, &[4.0, 1.0]).unwrap();
    let initial_logl = pip.cost(info);
    let o_pip = TopologyOptimiser::new(&pip, info).run().unwrap();
    assert!(o_pip.final_logl > initial_logl);
    assert_relative_eq!(o_pip.initial_logl, initial_logl);

    let model = DNASubstModel::new(K80, &[4.0, 1.0]).unwrap();
    let initial_logl = model.cost(info);
    let o_k80 = TopologyOptimiser::new(&model, info).run().unwrap();
    assert!(o_k80.final_logl > initial_logl);
    assert_relative_eq!(o_k80.initial_logl, initial_logl);
    assert_eq!(o_pip.i.tree.robinson_foulds(&o_k80.i.tree), 0);
}

#[test]
fn pip_vs_subst_protein_tree_nogaps() {
    // let info = PhyloInfoBuilder::new(PathBuf::from(
    //     "./data/phyml_protein_example/nogap_seqs.fasta",
    // ))
    // .build()
    // .unwrap();
    // let model = PIPModel::<ProteinSubstModel>::new(WAG, &[50.0, 0.1]).unwrap();
    // let unopt_logl = model.cost(&info);
    // let o = TopologyOptimiser::new(&model, &info).run().unwrap();
    // assert!(o.final_logl >= unopt_logl);
    // The above ran in 308.93s

    let pip_info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/phyml_protein_example/nogap_seqs.fasta"),
        PathBuf::from("./data/phyml_protein_example/optimisation_pip_nj_start.newick"),
    )
    .build()
    .unwrap();

    let subst_info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/phyml_protein_example/nogap_seqs.fasta"),
        PathBuf::from("./data/phyml_protein_example/optimisation_nj_start.newick"),
    )
    .build()
    .unwrap();
    // compare the tree created with a substitution model to the one with PIP
    // the trees are not identical, but very close
    assert!(pip_info.tree.robinson_foulds(&subst_info.tree) <= 2);
}

#[test]
fn protein_optimise_model_tree() {
    // let info = PhyloInfoBuilder::new(PathBuf::from("./data/phyml_protein_example/seqs.fasta"))
    //     .build()
    //     .unwrap();
    // let model = PIPModel::<ProteinSubstModel>::new(WAG, &[1.4, 0.5]).unwrap();
    // let unopt_logl = model.cost(&info);
    // let o = ModelOptimiser::new(&model, &info, Empirical).run().unwrap();
    // let model_opt_logl = o.final_logl;
    // assert!(model_opt_logl >= unopt_logl);
    // let o = TopologyOptimiser::new(&o.model, &info).run().unwrap();
    // assert!(model_opt_logl >= unopt_logl);
    // assert!(o.final_logl >= unopt_logl);
    // // The above ran in 703.87s

    // let info = PhyloInfoBuilder::new(PathBuf::from("./data/phyml_protein_example/seqs.fasta"))
    //     .build()
    //     .unwrap();
    // let model = PIPModel::<ProteinSubstModel>::new(WAG, &[1.4, 0.5]).unwrap();
    // let unopt_logl = model.cost(&info);
    // let o = TopologyOptimiser::new(&model, &info).run().unwrap();
    // assert!(o.final_logl >= unopt_logl);
    // The above ran in 497.22s

    let optim_model = PIPModel::<ProteinSubstModel>::new(WAG, &[49.56941, 0.09352]).unwrap();
    let with_model_optim = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/phyml_protein_example/seqs.fasta"),
        PathBuf::from(
            "./data/phyml_protein_example/optimisation_pip_nj_start_model_optim_gaps.newick",
        ),
    )
    .build()
    .unwrap();

    let model = ProteinSubstModel::new(WAG, &[1.4, 0.5]).unwrap();
    let wo_model_optim = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/phyml_protein_example/seqs.fasta"),
        PathBuf::from(
            "./data/phyml_protein_example/optimisation_pip_nj_start_model_optim_gaps.newick",
        ),
    )
    .build()
    .unwrap();

    assert!(with_model_optim.tree.robinson_foulds(&wo_model_optim.tree) == 0);
    assert!(optim_model.cost(&with_model_optim) > model.cost(&wo_model_optim));
}
