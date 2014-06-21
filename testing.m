% load( 'train.mat' ) ;
load( 'test.mat' ) ;
for i = 1 : size( feature , 1 )
    feature( i , : ) = feature( i , : ) / norm( feature( i , : ) ) ;
end
load( 'SVMs.mat' ) ;
feature_count = size( feature , 1 ) ;
label_count = 24 ;

label_predict_score = zeros( feature_count , label_count ) ;
for i = 1 : label_count
    [ ~ , ~ , etsi ] = svmpredict( ones( feature_count , 1 ) , feature , SVMs{ i } , '-q' ) ;
    label_predict_score( : , i ) = etsi ;
end
[ ~ , label_predict ] = max( label_predict_score' ) ;
label_predict = label_predict' ;
disp( sprintf( 'Test label accuracy: %.2f' , sum( label == label_predict ) / feature_count ) ) ;

label_map = zeros( label_count , 1 ) ;
for i = 1 : label_count 
    label_map( i ) = sum( and( label_predict == i , label_predict == label ) ) / ...
        sum( label == i ) ;
end
