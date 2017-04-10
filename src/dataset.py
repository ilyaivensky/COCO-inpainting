#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import numbers
import h5py
from scipy.sparse import csr_matrix
from fuel.datasets import H5PYDataset
from fuel.utils import Subset

import logging
from scipy.sparse.csgraph._min_spanning_tree import csr_matrix

class H5PYSparseDataset(H5PYDataset):    
    
    def __init__(self, file_or_path, which_sets, **kwargs):
        self.logger = logging.getLogger('H5PYSparseDataset')
        super(H5PYSparseDataset, self).__init__(file_or_path, which_sets, sort_indices=False, **kwargs)
     
    @staticmethod
    def get_axis_labels(h5file):
        """Returns axis labels for all sources in an HDF5 dataset.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.

        Returns
        -------
        axis_labels : dict
            Maps source names to a tuple of str representing the axis
            labels.

        """
        axis_labels = {}
        vlen_sources = H5PYSparseDataset.get_vlen_sources(h5file)
        for source_name in H5PYDataset.get_all_sources(h5file):
            if source_name in vlen_sources:
                axis_labels[source_name] = (
                    (h5file[source_name].dims[0].label,) +
                    tuple(label.decode('utf8') for label in
                          h5file[source_name].dims[0]['shape_labels']))
            else:
                if isinstance(h5file[source_name], h5py.Group):
                    continue
                axis_labels[source_name] = tuple(
                    dim.label for dim in h5file[source_name].dims)
        return axis_labels   
    
    @staticmethod
    def get_vlen_sources(h5file):
        """Returns the names of variable-length sources in an HDF5 dataset.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.
        split : str
            Name of the split.

        Returns
        -------
        vlen_sources : tuple of str
            Names of all variable-length sources in ``h5file``.

        """
        vlen_sources = []
        for source_name in H5PYDataset.get_all_sources(h5file):
            source = h5file[source_name]
            if isinstance(source, h5py.Group):
                continue
            if len(source.dims) > 0 and 'shapes' in source.dims[0]:
                if len(source.dims) > 1:
                    raise ValueError('Variable-length sources must have only '
                                     'one dimension.')
                vlen_sources.append(source_name)
        return vlen_sources
    
    @staticmethod
    def get_subsets(h5file, splits, sources):
        """Returns the subsets for a given splits/sources combination.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.
        splits : :class:`tuple` of :class:`str`
            Split names.
        sources : :class:`tuple` of :class:`str`
            Which sources should be considered.

        Returns
        -------
        :class:`list` of :class:`fuel.utils.Subset`
            The subsets, one per source in ``sources``, associated with
            the splits/sources combination.

        """
        subsets = [Subset.empty_subset(len(h5file[source_name]))
                    if isinstance(h5file[source_name], h5py.Dataset) else 
                   Subset.empty_subset(h5file[source_name].attrs['shape0']) for source_name in sources]
        
        for split in splits:
            for i, source in enumerate(sources):
                row, = [r for r in h5file.attrs['split'] if
                        (r['split'].decode('utf8') == split and
                         r['source'].decode('utf8') == source)]
                    
                if isinstance(h5file[source], h5py.Dataset):
                    if row['indices']:
                        subsets[i] += Subset(
                            h5file[row['indices']], len(h5file[source]))
                    else:
                        subsets[i] += Subset(
                            slice(row['start'], row['stop']), len(h5file[source]))
                
                else:
                    subsets[i] += Subset(
                            slice(row['start'], row['stop']), h5file[source].attrs['shape0'])

        return subsets
    
    def load(self):
        # If the dataset is unpickled, it makes no sense to have an external
        # file handle. However, since `load` is also called during the lifetime
        # of a dataset (e.g. if load_in_memory = True), we don't want to
        # accidentally overwrite the reference to a potential external file
        # handle, hence this check.
        if not hasattr(self, '_external_file_handle'):
            self.external_file_handle = None

        self._out_of_memory_open()
        handle = self._file_handle

        # Infer subsets based on `which_sets`
        subsets = self.get_subsets(handle, self.which_sets, self.sources)
        # Sanity check to make sure that all sources have equal length
        if any(subset.num_examples != subsets[0].num_examples for subset in
                subsets):
            raise ValueError("sources have different lengths")
        # Produce the final subsets by taking the `subset` constructor argument
        # into account.
        self.subsets = [Subset.subset_of(subset, self.user_given_subset)
                        for subset in subsets]

        # Load data sources and source shapes (if requested)
        if self.load_in_memory:
            data_sources = []
            source_shapes = []
            n_examples = None
            for source_name, subset in zip(self.sources, self.subsets):
                if isinstance(handle[source_name], h5py.Dataset):
                    data_sources.append(
                        subset.index_within_subset(
                            handle[source_name], slice(None)))
                    if source_name in self.vlen_sources:
                        shapes = subset.index_within_subset(
                            handle[source_name].dims[0]['shapes'],
                            slice(None))
                    else:
                        shapes = None
                    source_shapes.append(shapes)
                    n_examples = len(data_sources[-1])
                else:
                    data = handle['{}/data'.format(source_name)]
                    indices = handle['{}/indices'.format(source_name)]
                    indptr = handle['{}/indptr'.format(source_name)]
         
                    indexable = csr_matrix((data, indices, indptr), 
                                           shape=(handle[source_name].attrs['shape0'], 
                                                  handle[source_name].attrs['shape1']))
                    
                    data_sources.append(
                        subset.index_within_subset(
                            indexable, slice(None)))
                    
                    shapes = None
                    source_shapes.append(shapes)
                    n_examples = data_sources[-1].shape[0]
                           
            self.data_sources = tuple(data_sources)
            self.source_shapes = tuple(source_shapes)
            # This exists only for request sanity checking purposes.
            self.in_memory_subset = Subset(
                slice(None), n_examples)
            self.in_memory_sources = None
        else:
            self.in_memory_sources = OrderedDict()
            for source_name, subset in zip(self.sources, self.subsets):
                if not isinstance(handle[source_name], h5py.Dataset):
                    data = handle['{}/data'.format(source_name)]
                    indices = handle['{}/indices'.format(source_name)]
                    indptr = handle['{}/indptr'.format(source_name)]
         
                    indexable = csr_matrix((data, indices, indptr), 
                                           shape=(handle[source_name].attrs['shape0'], 
                                                  handle[source_name].attrs['shape1']))
                    
                    self.in_memory_sources[source_name] = subset.index_within_subset(indexable, slice(None))
                    
            self.data_sources = None
            self.source_shapes = None
            self.in_memory_subset = Subset(
                slice(None), self.in_memory_sources.items()[0][1].shape[0])

        self._out_of_memory_close()
        
    def get_data(self, state=None, request=None):
        if self.load_in_memory:
            data, shapes = self._in_memory_get_data(state, request)
        else:
            data, shapes = self._out_of_memory_get_data(state, request)
        for i in range(len(data)):
            if shapes[i] is not None:
                if isinstance(request, numbers.Integral):
                    data[i] = data[i].reshape(shapes[i])
                else:
                    for j in range(len(data[i])):
                        data[i][j] = data[i][j].reshape(shapes[i][j])
        return tuple(data)
    
    def _in_memory_get_data(self, state=None, request=None):
        if state is not None or request is None:
            raise ValueError
            
        data = [self.in_memory_subset.index_within_subset(data_source, request)
                if not isinstance(data_source, csr_matrix) 
                else data_source[request,:].toarray()
                for data_source in self.data_sources]
        shapes = [self.in_memory_subset.index_within_subset(shape, request)
                  if shape is not None else None
                  for shape in self.source_shapes]
        return data, shapes
     
    def _out_of_memory_get_data(self, state=None, request=None):
        if not isinstance(request, (numbers.Integral, slice, list)):
            raise ValueError()
        data = []
        shapes = []
        # TODO: This is not an ideal solution, really unpickling should be
        # restoring the state of the dataset fully, i.e. load() must be
        # modified to account for the out-of-memory case.
        # See https://git.io/vKkSm
        try:
            handle = self._file_handle
        except IOError:
            self._out_of_memory_open()
            handle = self._file_handle
        for source_name, subset in zip(self.sources, self.subsets):
            # Process the data request within the context of the data source
            # subset
            if isinstance(handle[source_name], h5py.Dataset):
                data.append(
                    subset.index_within_subset(
                        handle[source_name], request))
                # If this source has variable length, get the shapes as well
                if source_name in self.vlen_sources:
                    shapes.append(
                        subset.index_within_subset(
                            handle[source_name].dims[0]['shapes'], request))
                else:
                    shapes.append(None)
            else:
                data.append(
                    self.in_memory_sources[source_name][request,:].toarray())
                shapes.append(None)
        return data, shapes
    