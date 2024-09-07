"""_simulation_models.py_
Define the model for simulating different types of data
Total four types of data: non-hybrid, hybrid, introgression, introgression_with_gene_flow

The parameters used in the models are referenced in the following work:
Blischak, P. D., Barker, M. S., & Gutenkunst, R. N. (2021).
Chromosome-scale inference of hybrid speciation and admixture with convolutional neural networks.
Molecular ecology resources, 21(8), 2676-2688. https://doi.org/10.1111/1755-0998.13355

The parameters in mutation simulation part are reference in the following work:
Blischak, P. D., Chifman, J., Wolfe, A. D., & Kubatko, L. S. (2018).
HyDe: A Python Package for Genome-Scale Hybridization Detection.
Systematic biology, 67(5), 821-829. https://doi.org/10.1093/sysbio/syy023

"""

# import packages
import msprime
import numpy as np
import matplotlib.pyplot as plt
import tskit
# import IPython.display as display
import os
import sys
import demesdraw
from tabulate import tabulate
import pandas as pd


# define no hybridization function


def no_hybrid(coal_units=1.0):
    """
    Out    P3    P2    P1
     |      |     |     |
     |      |     |     |
     |      |     |-----|
     |      |           |
     |      |           |
     |      |-----------|
     |                  |
     |                  |
     |                  |
     |                  |
     |------------------|
               |

    """
    Ne = 1000.0
    time_units = 2.0 * Ne * coal_units  # effective population size
    #length = np.random.choice([x for x in range(int(1e7), int(5e7)+1, 10000)])
    length = 2e5
    # Draw divergence times
    t1 = np.random.gamma(10.0, 1.0 / 10.0)
    t2 = np.random.gamma(10.0, 1.0 / 10.0) + t1
    t3 = np.random.gamma(20.0, 1.0 / 10.0) + t2

    # Draw recombination and mutation rates
    recomb_rate = np.random.uniform(1.25e-7, 6.25e-6)#(2.5e-9, 2.5e-8)
    mutate_rate = np.random.uniform(1.25e-7, 6.25e-6)#(2.5e-9, 2.5e-8)

    # Set up sampling sizes, 5 individuals from each population except for the outgroup
    samples = [
        msprime.SampleSet(1, ploidy=1, population="Out"),
        msprime.SampleSet(5, ploidy=1, population="P1"),
        msprime.SampleSet(5, ploidy=1, population="P2"),
        msprime.SampleSet(5, ploidy=1, population="P3"),
    ]

    # Set up demography events
    # Define four populations, each has an initial sample size of 1000
    demography = msprime.Demography()
    demography.add_population(name="Out", initial_size=Ne)
    demography.add_population(name="P1", initial_size=Ne)
    demography.add_population(name="P2", initial_size=Ne)
    demography.add_population(name="P3", initial_size=Ne)

    # Add intermediate populations, which are the most recent common ancestors before each divergence
    demography.add_population(name="P12", initial_size=Ne)
    demography.add_population(name="P123", initial_size=Ne)
    demography.add_population(name="P123Out", initial_size=Ne)

    # Set up divergence events
    # from (P1P2) common ancestor to P1 and P2
    demography.add_population_split(
        time=t1 * time_units, derived=["P1", "P2"],
        ancestral="P12"
    )
    # from (P1P2P3) common ancestor to P12 and P3
    demography.add_population_split(
        time=t2 * time_units, derived=["P12", "P3"],
        ancestral="P123"
    )
    # from (P1P2P3P4) common ancestor to P123 and Outgroup
    demography.add_population_split(
        time=t3 * time_units, derived=["P123", "Out"],
        ancestral="P123Out"
    )

    # Use DemographyDebugger to visualize the demographic model
    debugger = msprime.DemographyDebugger(demography=demography)
    debugger.print_history()

    demesdraw.tubes(demography.to_demes())

    ts = msprime.sim_ancestry(
        recombination_rate=recomb_rate,
        sequence_length=length,
        samples=samples,
        demography=demography,
        record_migrations=True,
    )

    ts = msprime.sim_mutations(
        ts,
        rate=mutate_rate,
        model=msprime.GTR(
            # “A<>C”, “A<>G”, “A<>T”, “C<>G”, “C<>T”, “G<>T”
            relative_rates=[1.0, 0.2, 10.0, 0.75, 3.2, 1.6],
            equilibrium_frequencies=[0.15, 0.35,
                                     0.15, 0.35],  # A C G T
        ))

    # Print out the configurations
    config_data = [
        ["Coalescent Units", f"{coal_units}"],
        ["Time Units", f"{time_units}"],
        ["Sequence Length", f"{length}"],
        [f"t1={t1:.2f} and diverge time", f"{t1*time_units}"],
        [f"t2={t2:.2f} and diverge time", f"{t2*time_units}"],
        [f"t3={t3:.2f} and diverge time", f"{t3*time_units}"],
        ["Recombination Rate", f"{recomb_rate:.4e}"],
        ["Mutation Rate", f"{mutate_rate:.4e}"]
    ]

    print("\n No-Hybridization Simulation Configurations:")
    print(tabulate(config_data, headers=[
          "Parameter", "Value"], tablefmt="grid"))

    return ([time_units, length, t1, t2, t3, recomb_rate, mutate_rate],
            ts)


# define hybridization function
# hybridization P1, P3 are parents of P2 with ratio sum up to 1

def hybrid(coal_units=1.0):
    """
    Out    P3    P2    P1
     |      |     |     |
     |      |     |     |
     |      |----/ \----|
     |      |           |
     |      |           |
     |      |-----------|
     |                  |
     |                  |
     |                  |
     |                  |
     |------------------|
               |

    """
    Ne = 1000.0
    time_units = 2.0 * Ne * coal_units
    #length = np.random.choice([x for x in range(int(1e7), int(5e7)+1, 10000)])
    length = 2e5
    # Draw divergence times
    t1 = np.random.gamma(10.0, 1.0 / 10.0)
    t2 = np.random.gamma(10.0, 1.0 / 10.0) + t1
    t3 = np.random.gamma(20.0, 1.0 / 10.0) + t2

    # Draw recombination and mutation rates
    recomb_rate = np.random.uniform(1.25e-7, 6.25e-6)#(2.5e-9, 2.5e-8)
    mutate_rate = np.random.uniform(1.25e-7, 6.25e-6)#(2.5e-9, 2.5e-8)

    # Draw hybridization fraction
    gamma_1 = np.random.uniform(0.25, 0.75)  # P1 proportion
    gamma_2 = 1 - gamma_1  # P3 proportion

    # Set up sampling sizes, 5 individuals from each population except for the outgroup
    samples = [
        msprime.SampleSet(1, ploidy=1, population="Out"),
        msprime.SampleSet(5, ploidy=1, population="P1"),
        msprime.SampleSet(5, ploidy=1, population="P2"),
        msprime.SampleSet(5, ploidy=1, population="P3"),
    ]

    # Set up demography events
    # Define four populations, each has an initial sample size of 1000
    demography = msprime.Demography()
    demography.add_population(name="Out", initial_size=Ne)
    demography.add_population(name="P1", initial_size=Ne)
    demography.add_population(name="P2", initial_size=Ne)
    demography.add_population(name="P3", initial_size=Ne)

    # Add intermediate populations, which are the most recent common ancestors before each divergence
    demography.add_population(name="P13", initial_size=Ne)
    demography.add_population(name="P13Out", initial_size=Ne)

    # Divergence events
    # P1 and P3 have hybrid child P2 at t1
    demography.add_admixture(
        time=t1 * time_units,
        derived="P2",
        ancestral=["P1", "P3"],
        proportions=[gamma_1, gamma_2]
    )

    # from (P1P3) common ancestor to P1 and P3
    demography.add_population_split(
        time=t2 * time_units, derived=["P1", "P3"],
        ancestral="P13"
    )
    # from (P1P3out) common ancestor to P123 and Outgroup
    demography.add_population_split(
        time=t3 * time_units, derived=["P13", "Out"],
        ancestral="P13Out"
    )

    # Use DemographyDebugger to visualize the demographic model
    debugger = msprime.DemographyDebugger(demography=demography)
    debugger.print_history()

    demesdraw.tubes(demography.to_demes())

    ts = msprime.sim_ancestry(
        recombination_rate=recomb_rate,
        sequence_length=length,
        samples=samples,
        demography=demography,
        record_migrations=True,
    )

    ts = msprime.sim_mutations(
        ts,
        rate=mutate_rate,
        model=msprime.GTR(
            # “A<>C”, “A<>G”, “A<>T”, “C<>G”, “C<>T”, “G<>T”
            relative_rates=[1.0, 0.2, 10.0, 0.75, 3.2, 1.6],
            equilibrium_frequencies=[0.15, 0.35,
                                     0.15, 0.35],  # A C G T
        ))

    # Print out the configurations in a table
    config_data = [
        ["Coalescent Units", f"{coal_units}"],
        ["Time Units", f"{time_units}"],
        ["Sequence Length", f"{length}"],
        [f"t1={t1:.2f} and diverge time", f"{t1*time_units}"],
        [f"t2={t2:.2f} and diverge time", f"{t2*time_units}"],
        [f"t3={t3:.2f} and diverge time", f"{t3*time_units}"],
        ["Recombination Rate", f"{recomb_rate:.4e}"],
        ["Mutation Rate", f"{mutate_rate:.4e}"],
        ["P2 from P1 proportion", f"{gamma_1:4f}"],
        ["P2 from P3 proportion", f"{gamma_2:4f}"]
    ]

    print("\nHybridization Simulation Configurations:")
    print(tabulate(config_data, headers=[
          "Parameter", "Value"], tablefmt="grid"))

    return ([time_units, length, t1, t2, t3, recomb_rate, mutate_rate, gamma_1, gamma_2],
            ts)


# Define introgression function
# P3, as an outgroup of P1 and P2, there is a gene flow from P3 to P2 after P1P2 divergence

def introgression(coal_units=1.0):
    """
    Out    P3    P2    P1
     |      |     |     |
     |      |---->|     |
     |      |     |-----|
     |      |           |
     |      |           |
     |      |-----------|
     |                  |
     |                  |
     |                  |
     |                  |
     |------------------|
               |

    """
    Ne = 1000.0
    time_units = 2.0 * Ne * coal_units
    #length = np.random.choice([x for x in range(int(1e7), int(5e7)+1, 10000)])
    length = 2e5
    # Draw divergence times
    t1 = np.random.gamma(10.0, 1.0 / 10.0)
    t2 = np.random.gamma(10.0, 1.0 / 10.0) + t1
    t3 = np.random.gamma(20.0, 1.0 / 10.0) + t2

    # Draw recombination and mutation rates
    recomb_rate = np.random.uniform(1.25e-7, 6.25e-6)#(2.5e-9, 2.5e-8), increase by 50 times
    mutate_rate = np.random.uniform(1.25e-7, 6.25e-6)#(2.5e-9, 2.5e-8)

    # Draw P3 introgression fraction
    intro_ratio = np.random.uniform(0.01, 0.25)
    intro_time = np.random.uniform(0.1 * t1, 0.9 * t1)  # introgression time t0

    # Set up sampling sizes, 5 individuals from each population except for the outgroup
    samples = [
        msprime.SampleSet(1, ploidy=1, population="Out"),
        msprime.SampleSet(5, ploidy=1, population="P1"),
        msprime.SampleSet(5, ploidy=1, population="P2"),
        msprime.SampleSet(5, ploidy=1, population="P3"),
    ]

    # set up demography events
    # define four populations, each has an initial sample size of 1000
    demography = msprime.Demography()
    demography.add_population(name="Out", initial_size=Ne)
    demography.add_population(name="P1", initial_size=Ne)
    demography.add_population(name="P2", initial_size=Ne)
    demography.add_population(name="P3", initial_size=Ne)

    # Add intermediate populations, which are the most recent common ancestors before each divergence
    demography.add_population(name="P12", initial_size=Ne)
    demography.add_population(name="P123", initial_size=Ne)
    demography.add_population(name="P123Out", initial_size=Ne)

    # Admixture and divergence events
    # P2 receives gene flow from P3
    demography.add_mass_migration(
        time=intro_time * time_units,
        source="P2",
        dest="P3",
        proportion=intro_ratio
    )

    # from (P1P2) common ancestor to P1 and P2
    demography.add_population_split(
        time=t1 * time_units, derived=["P1", "P2"],
        ancestral="P12"
    )

    # from (P1P2)P3 common ancestor to P12 and P3
    demography.add_population_split(
        time=t2 * time_units, derived=["P12", "P3"],
        ancestral="P123"
    )
    # from (P1P2)P3)Out common ancestor to P123 and Outgroup
    demography.add_population_split(
        time=t3 * time_units, derived=["P123", "Out"],
        ancestral="P123Out"
    )

    # Use DemographyDebugger to visualize the demographic model
    debugger = msprime.DemographyDebugger(demography=demography)
    debugger.print_history()

    demesdraw.tubes(demography.to_demes())

    ts = msprime.sim_ancestry(
        recombination_rate=recomb_rate,
        sequence_length=length,
        samples=samples,
        demography=demography,
        record_migrations=True,
    )

    ts = msprime.sim_mutations(
        ts,
        rate=mutate_rate,
        model=msprime.GTR(
            # “A<>C”, “A<>G”, “A<>T”, “C<>G”, “C<>T”, “G<>T”.
            relative_rates=[1.0, 0.2, 10.0, 0.75, 3.2, 1.6],
            equilibrium_frequencies=[0.15, 0.35,
                                     0.15, 0.35],  # A C G T
        ))

    # Print out the configurations in a table
    config_data = [
        ["Coalescent Units", f"{coal_units}"],
        ["Time Units", f"{time_units}"],
        ["Sequence Length", f"{length}"],
        [f"t1={t1:.2f} and diverge time", f"{t1*time_units}"],
        [f"t2={t2:.2f} and diverge time", f"{t2*time_units}"],
        [f"t3={t3:.2f} and diverge time", f"{t3*time_units}"],
        ["Recombination Rate", f"{recomb_rate:4e}"],
        ["Mutation Rate", f"{mutate_rate:4e}"],
        [f"P3->P2 introgression time t0={intro_time:.2f}",
            f"{intro_time*time_units}"],
        ["P3->P2 introgression ratio", f"{intro_ratio:4f}"]
    ]

    print("\nIntrogression Simulation Configurations:")
    print(tabulate(config_data, headers=[
          "Parameter", "Value"], tablefmt="grid"))

    return ([time_units, length, t1, t2, t3, recomb_rate, mutate_rate, intro_time, intro_ratio],
            ts)


# define intro with gene flow function
# P3, as an outgroup of P1 and P2, there is a gene flow from P3 to P2 after P1P2 divergence
# also, there is a continuous gene flow between P1 and P2

def introgression_w_gflow(coal_units=1.0):
    """
    Out    P3    P2    P1
     |      |     |<--->|
     |      |---->|<--->|
     |      |     |-----|
     |      |           |
     |      |           |
     |      |-----------|
     |                  |
     |                  |
     |                  |
     |                  |
     |------------------|
               |

    """
    Ne = 1000.0
    time_units = 2.0 * Ne * coal_units
    #length = np.random.choice([x for x in range(int(1e7), int(5e7)+1, 10000)])
    length = 2e5
    # Draw divergence times
    t1 = np.random.gamma(10.0, 1.0 / 10.0)
    t2 = np.random.gamma(10.0, 1.0 / 10.0) + t1
    t3 = np.random.gamma(20.0, 1.0 / 10.0) + t2

    # Draw recombination an mutation rates
    recomb_rate = np.random.uniform(1.25e-7, 6.25e-6)#(2.5e-9, 2.5e-8)
    mutate_rate = np.random.uniform(1.25e-7, 6.25e-6)#(2.5e-9, 2.5e-8)

    # Draw P3 introgression fraction
    intro_ratio = np.random.uniform(0.01, 0.25)
    intro_time = np.random.uniform(0.1 * t1, 0.9 * t1)  # introgression time t0

    # Set up the constant migration rate between P1 and P2
    m_rate = np.random.uniform(2.5e-4, 5e-4)

    # Set up sampling sizes, 5 individuals from each population except for the outgroup
    samples = [
        msprime.SampleSet(1, ploidy=1, population="Out"),
        msprime.SampleSet(5, ploidy=1, population="P1"),
        msprime.SampleSet(5, ploidy=1, population="P2"),
        msprime.SampleSet(5, ploidy=1, population="P3"),
    ]

    # Set up demography events

    # define four populations, each has an initial sample size of 1000
    demography = msprime.Demography()
    demography.add_population(name="Out", initial_size=Ne)
    demography.add_population(name="P1", initial_size=Ne)
    demography.add_population(name="P2", initial_size=Ne)
    demography.add_population(name="P3", initial_size=Ne)

    # Add intermediate populations, which are the most recent common ancestors before each divergence
    demography.add_population(name="P12", initial_size=Ne)
    demography.add_population(name="P123", initial_size=Ne)
    demography.add_population(name="P123Out", initial_size=Ne)

    # Admixture and divergence events

    # P1 and P2 have constant mass migration
    demography.set_symmetric_migration_rate(
        populations=["P1", "P2"],
        rate=m_rate
    )

    # P2 receives one gene flow from P3
    demography.add_mass_migration(
        time=intro_time * time_units,
        source="P2",
        dest="P3",
        proportion=intro_ratio
    )

    # stops the gene flow between P1 and P2
    demography.add_symmetric_migration_rate_change(
        time=t1 * time_units,
        populations=["P1", "P2"],
        rate=0
    )

    # from (P1P2) common ancestor to P1 and P2
    demography.add_population_split(
        time=t1 * time_units, derived=["P1", "P2"],
        ancestral="P12"
    )

    # from (P1P2)P3 common ancestor to P12 and P3
    demography.add_population_split(
        time=t2 * time_units, derived=["P12", "P3"],
        ancestral="P123"
    )
    # from (P1P2)P3)Out common ancestor to P123 and Outgroup
    demography.add_population_split(
        time=t3 * time_units, derived=["P123", "Out"],
        ancestral="P123Out"
    )

    # Use DemographyDebugger to visualize the demographic model
    debugger = msprime.DemographyDebugger(demography=demography)
    debugger.print_history()

    demesdraw.tubes(demography.to_demes())

    ts = msprime.sim_ancestry(
        recombination_rate=recomb_rate,
        sequence_length=length,
        samples=samples,
        demography=demography,
        record_migrations=True,
    )

    ts = msprime.sim_mutations(
        ts,
        rate=mutate_rate,
        model=msprime.GTR(
            # “A<>C”, “A<>G”, “A<>T”, “C<>G”, “C<>T”, “G<>T”.
            relative_rates=[1.0, 0.2, 10.0, 0.75, 3.2, 1.6],
            equilibrium_frequencies=[0.15, 0.35,
                                     0.15, 0.35],  # A C G T
        ))

    # Print out the configurations in a table
    config_data = [
        ["Coalescent Units", f"{coal_units}"],
        ["Time Units", f"{time_units}"],
        ["Sequence Length", f"{length}"],
        [f"t1={t1:.2f} and diverge time", f"{t1*time_units}"],
        [f"t2={t2:.2f} and diverge time", f"{t2*time_units}"],
        [f"t3={t3:.2f} and diverge time", f"{t3*time_units}"],
        ["Recombination Rate", f"{recomb_rate:4e}"],
        ["Mutation Rate", f"{mutate_rate:4e}"],
        [f"P3->P2 introgression time t0={intro_time:.2f}",
            f"{intro_time*time_units}"],
        ["P3->P2 introgression ratio", f"{intro_ratio:4f}"],
        ["P1<->P2 gene flow rate", f"{m_rate:8f}"]
    ]

    print("\nIntrogression with gene flow Simulation Configurations:")
    print(tabulate(config_data, headers=[
          "Parameter", "Value"], tablefmt="grid"))

    return ([time_units, length, t1, t2, t3, recomb_rate, mutate_rate, intro_time, intro_ratio, m_rate],
            ts)
