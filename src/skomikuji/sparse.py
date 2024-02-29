from scipy.sparse import csr_array, csr_matrix, lil_matrix, lil_array, spmatrix, sparray

def sparse_to_feature_value_pairs(B: spmatrix | sparray):
    """
    Converts a sparse matrix into the list of list format where rows are unrolled in tuples of pairs (col_index, value).

    Parameters
    ----------
    B:
    For example:

    D = csr_matrix(
        np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1]
            ]
        )
        )
    # defines the matrix in list of lists format
    L = D.tolil()

    The function
    (
        # first row
        (
            (1, 1),  # column 1, value 1
        ),
        # second row
        (),
        # third row
        (
            (0, 1), # column 0, value 1
        ),
        # fourth row
        (
            (0, 1), # column 0, value 1
            (3, 1) # column 3, value 1
        )
    )
    """
    if not isinstance(B, (lil_matrix, lil_array)):
        # we require LIL format
        L: lil_matrix = B.tolil()
    # then reshape the data
    return tuple(
        tuple((_r, _d) for _r, _d in zip(r, d)) for r, d in zip(L.rows, L.data)
    )