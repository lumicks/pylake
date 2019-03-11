function out = to_matlab(pyarray)
% TO_MATLAB Convert the given Python array into a Matlab array

% Python uses row-major ordering (C style) 
% and Matlab uses column-major (Fortran style)
fortran_array = py.numpy.asfortranarray(pyarray);
out = double(py.array.array('d', py.numpy.nditer(fortran_array)));

if fortran_array.ndim ~= 1
    shape = int64(py.array.array('q', fortran_array.shape));
    out = reshape(out, shape);
end

end
