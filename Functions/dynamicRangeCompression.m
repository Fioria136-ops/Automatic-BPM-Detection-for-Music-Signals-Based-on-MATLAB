function x_out = dynamicRangeCompression(x, compressionThreshold, compressionRatio)
%Initialize output signal
x_out = x;
x_abs = abs(x);
%Find the part that needs to be compressed (the part that exceeds the threshold)
aboveThreshold = x_abs > compressionThreshold;

%Apply compression to the parts that exceed the threshold, only changing the parts that need to be compressed
x_out(aboveThreshold) = sign(x(aboveThreshold)) .* ...
     (compressionThreshold + (x_abs(aboveThreshold) - compressionThreshold) / compressionRatio);

end
