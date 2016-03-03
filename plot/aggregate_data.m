function [ agg_vec ] = aggregate_data( vec, groupsize )
    %AGGREGATE_DATA aggregates a vector by specified group size. Coverts cell to matrix if need be
    
    if( iscell(vec) ); vec = cell2mat(vec); end
    
    agg_vec = sum(  reshape(vec, [(numel(vec)/groupsize) groupsize])  );

end

