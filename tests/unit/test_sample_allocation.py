import os.path
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

from mxmc.sample_allocation import *


@pytest.fixture
def compressed_allocation():
    return np.array([[1, 1, 1, 1, 0, 0],
                     [5, 0, 1, 1, 1, 1],
                     [10, 0, 0, 0, 1, 1]])

@pytest.fixture
def sample_allocation(compressed_allocation):
    warnings.filterwarnings("ignore",
                            message="Allocation Warning",
                            category=UserWarning)
    return SampleAllocation(compressed_allocation, 'MFMC')


@pytest.fixture
def saved_allocation_path(sample_allocation, tmp_path):
    p = tmp_path / "test_allocation.h5"
    path_str = str(p)
    sample_allocation.save(path_str)
    return path_str


@pytest.fixture
def expanded_allocation():
    return [[1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1]]


@pytest.fixture
def expanded_allocation_dataframe(expanded_allocation):
    return pd.DataFrame(columns=['0', '1_1', '1_2', '2_1', '2_2'],
                        data=expanded_allocation)


@pytest.fixture
def input_names():
    return ['input1', 'input2', 'input3']


@pytest.fixture
def input_array():
    return np.array([[1, 2, 3],
                     [2, 3, 4],
                     [3, 4, 5],
                     [6.4, 3, 7],
                     [2.4, 34.5, 54],
                     [12, 3, 4],
                     [2, 13, 4],
                     [2, 38, 14],
                     [2.4, 83, 4.1],
                     [28, 3.4, 4],
                     [24, 3, 4],
                     [2.8, 43, 4],
                     [2, 3.7, 44],
                     [42, 37, 74],
                     [72, 4.3, 4],
                     [27, 3, 9.4]])


def test_compressed_allocation(sample_allocation, compressed_allocation):
    assert np.array_equal(sample_allocation.compressed_allocation,
                          compressed_allocation)


def test_num_models(sample_allocation):
    assert sample_allocation.num_models == 3


def test_get_column_names(sample_allocation):
    assert sample_allocation._get_column_names() == ['0', '1_1', '1_2', '2_1',
                                                     '2_2']


def test_one_model_num_models():
    test_allocation = SampleAllocation(np.array([[10, 1]]), 'MFMC')
    assert test_allocation.num_models == 1


def test_expand_allocation(sample_allocation, expanded_allocation_dataframe):
    assert pd.DataFrame.equals(sample_allocation.expanded_allocation,
                               expanded_allocation_dataframe)


def test_get_number_of_samples_per_model(sample_allocation):
    assert np.array_equal(sample_allocation.get_number_of_samples_per_model(),
                          np.array([1, 6, 15]))

def test_get_number_of_samples_per_model_MC():
    compressed_allocation = np.array([[10, 1, 0, 0, 0, 0]])
    sample_allocation = SampleAllocation(compressed_allocation, 'MC') 
    assert np.array_equal(sample_allocation.get_number_of_samples_per_model(),
                          np.array([10, 0, 0]))

@pytest.mark.parametrize("model_num, expected_indices",
                         [(0, np.array([0])),
                          (1, np.arange(6)),
                          (2, np.arange(1, 16))])
def test_get_sample_indices_for_model_2(sample_allocation, model_num,
                                        expected_indices):
    model_indices = sample_allocation.get_sample_indices_for_model(model_num)
    np.testing.assert_array_equal(model_indices, expected_indices)


def test_get_total_number_of_samples(sample_allocation):
    assert sample_allocation.num_total_samples == 16


def test_h5_file_exists(saved_allocation_path):
    assert os.path.exists(saved_allocation_path)


def test_sample_allocation_read(saved_allocation_path, sample_allocation):
    loaded_allocation = read_allocation(saved_allocation_path)
    np.testing.assert_array_equal(loaded_allocation.compressed_allocation,
                                  sample_allocation.compressed_allocation)
    assert loaded_allocation.num_models == sample_allocation.num_models
    assert loaded_allocation.method == sample_allocation.method


def test_allocate_samples_to_models_not_enough_samples_error(sample_allocation):

    too_few_inputs = np.random.rand(10, 3)
    with pytest.raises(ValueError):
        _ = sample_allocation.allocate_samples_to_models(too_few_inputs)


def test_allocate_samples_to_models(sample_allocation, input_array):

    inputs_0 = input_array[np.arange(0, 1), :]
    inputs_1 = input_array[np.arange(0, 6), :]
    inputs_2 = input_array[np.arange(1, 16), :]
    ref_samples = [inputs_0, inputs_1, inputs_2]

    gen_samples = sample_allocation.allocate_samples_to_models(input_array)

    for i, gen_samples_i in enumerate(gen_samples):
        assert np.array_equal(ref_samples[i], gen_samples_i)
