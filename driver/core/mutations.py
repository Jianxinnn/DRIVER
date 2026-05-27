import logging


logger = logging.getLogger(__name__)


def find_mutations(wt, mt, alphabet):
    mutations = []
    for i in range(len(wt)):
        if wt[i] != mt[i]:
            mutations.append((i, alphabet.index(mt[i]), alphabet.index(wt[i]), f'{wt[i]}{i+1}{mt[i]}'))
    return mutations

def find_mutant_for_prosst(wt_seq: str, variant: str, offset_idx: int = 1):
    assert len(wt_seq) == len(variant), "Length must be the same."
    mutant = []
    for i in range(len(wt_seq)):
        if wt_seq[i] != variant[i]:
            mutant.append(f"{wt_seq[i]}{i + offset_idx}{variant[i]}")

    return ":".join(mutant)

def generate_mutations(wt, amino_acids='ACDEFGHIKLMNPQRSTVWY', mut_pos=None, mut_pos_info=None, group_muts=False):
    '''
    Generate single mutants or grouped mutants.

    mut_pos: optional list of positions to mutate.
    mut_pos_info: optional list of mutation labels, such as ['A1C', 'A2C'].
    group_muts: when True, generate all valid amino-acid combinations at mut_pos.

    
    '''
    mutations = []
    mut_info = []
    
    if mut_pos_info:
        logger.info("Generating mutations from provided mutation labels")
        # Generate variants directly from provided mutation labels.
        mut_pos_info = set(mut_pos_info)
        for mut in mut_pos_info:
            original_aa, position, new_aa = mut[0], int(mut[1:-1]), mut[-1]
            # Verify that mutation labels match the wild-type sequence.
            if wt[position - 1] == original_aa:
                mutant = wt[:position - 1] + new_aa + wt[position:]
                mutations.append(mutant)
                mut_info.append(mut)
            else:
                logger.warning("Mutation label %s does not match the wild-type sequence", mut)
    else:
        if group_muts and mut_pos is not None:
            logger.info("Generating group mutations at positions: %s", mut_pos)
            # Generate all combinatorial mutations.
            from itertools import product
            mut_positions = [(pos - 1) for pos in mut_pos]
            valid_combinations = []

            # Keep only combinations where every selected position changes.
            for combination in product(amino_acids, repeat=len(mut_positions)):
                if all(wt[idx] != new_aa for idx, new_aa in zip(mut_positions, combination)) and any(wt[idx] != new_aa for idx, new_aa in zip(mut_positions, combination)):
                    valid_combinations.append(combination)

            for combination in valid_combinations:
                mutant = list(wt)
                mut_description = []
                
                for idx, new_aa in zip(mut_positions, combination):
                    mutant[idx] = new_aa
                    mut_description.append(f'{wt[idx]}{idx + 1}{new_aa}')
                
                mutations.append(''.join(mutant))
                mut_info.append(','.join(mut_description))
        else:
            if mut_pos is not None:
                logger.info("Generating single-point mutations at positions: %s", mut_pos)
            else:
                logger.info("Generating all single mutations")

            # Generate all possible single mutants when no mutation labels are provided.
            for i in range(len(wt)):
                # Skip positions not in mut_pos if mut_pos is provided
                if mut_pos is not None and (i + 1) not in mut_pos:
                    continue

                for aa in amino_acids:
                    # Only generate a mutation if the current amino acid is different from the wild type
                    if wt[i] != aa:
                        mutant = wt[:i] + aa + wt[i + 1:]
                        mutations.append(mutant)
                        mut_info.append(f'{wt[i]}{i + 1}{aa}')

    return mutations, mut_info

def select_closest_values(results, esm2_results, vaspa_results, select_num=9):
    selected_results = []
    selected_esm2_results = []
    selected_vaspa_results = []

    for result, esm2_result, vaspa_result in zip(results, esm2_results, vaspa_results):
        diffs = [(abs(val - result[0]), idx) for idx, val in enumerate(result)]
        diffs.sort()
        selected_indices = [idx for _, idx in diffs[:select_num]]
        selected_result = [result[idx] for idx in selected_indices]
        selected_esm2_result = [esm2_result[idx] for idx in selected_indices]
        selected_vaspa_result = [vaspa_result[idx] for idx in selected_indices]
        selected_results.append(selected_result)
        selected_esm2_results.append(selected_esm2_result)
        selected_vaspa_results.append(selected_vaspa_result)
    return selected_results, selected_esm2_results, selected_vaspa_results

def select_for_covery(results, select_num=9):
    select_results = []
    for result in results:
        select_results.append(result[:select_num])
    return select_results
