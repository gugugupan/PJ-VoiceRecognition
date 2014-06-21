load( 'train.mat' ) ;
for i = 1 : size( feature , 1 )
    feature( i , : ) = feature( i , : ) / norm( feature( i , : ) ) ;
end
feature_count = size( feature , 1 ) ;
label_count = 24 ;

SVMs = cell( label_count , 1 ) ;
for i = 1 : label_count 
    positive_set = find( label == i ) ;
    negative_set = find( label ~= i ) ;
    train_feature = [ feature( positive_set , : ) ; feature( negative_set , : ) ] ;
    train_label = [ 1 * ones( length( positive_set ) , 1 ) ; -1 * ...
        ones( length( negative_set ) , 1 ) ] ;
    
    SVMs{ i } = svmtrain( train_label , train_feature , '-t 0 -w1 5 -q' ) ;
    [ lll , ~ , ccc ] = svmpredict( train_label , train_feature , SVMs{ i } , '-q' ) ;
%     disp( sprintf( '%d %d' , sum( and( lll == 1 , lll == train_label ) ) , ...
%         sum( and( lll == -1 , lll == train_label ) ) ) ) ;
end
save( 'SVMs.mat' , 'SVMs' ) ;