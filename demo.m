clear ;
label_list = { '����' , '��ˮ' , '����' , '����' , '����' , '����' , '����' , ...
    'С��' , 'â��' , '����' , 'С��' , '����' , '����' , '����' , '��¶' , ...
    '���' , '��¶' , '˪��' , '����' , 'Сѩ' , '��ѩ' , '����' , 'С��' , ...
    '��' , 'û�ҵ�' } ;

% Record your voice for 5 seconds.
recObj = audiorecorder( 44100 , 16 , 2 ) ;
disp( 'Start speaking: ' ) ;
recordblocking(recObj, 1.5 );
disp( 'End of Recording.' ) ;
audiowrite( 'test.wav' , getaudiodata( recObj ) , 44100 ) ;

load( 'train.mat' ) ;
for i = 1 : size( feature , 1 )
    feature( i , : ) = feature( i , : ) / norm( feature( i , : ) ) ;
end
load( 'SVMs.mat' ) ;
load( 'codebook.mat' ) ;
label_count = 24 ;

mfcc_feature = mfcc( 'test.wav' ) ;
test_feature = zeros( 1 , codebook_size ) ;
dist = zeros( codebook_size  , 1 ) ;
for i = 1 : size( mfcc_feature , 1 )
    for j = 1 : codebook_size
        dist( j ) = norm( mfcc_feature( i , : ) - codebook( j , : ) ) ;
    end
    [ ~ , k ] = min( dist ) ;
    test_feature( k ) = test_feature( k ) + 1 ;
end
test_feature = test_feature / norm( test_feature ) ;

label_predict_score = zeros( 1 , label_count ) ;
for i = 1 : label_count 
    [ ~ , ~ , label_predict_score( i ) ] = svmpredict( 1 , test_feature , SVMs{ i } , '-q' ) ;
end
[ p , l ] = max( label_predict_score ) ;
if ( p < 0 )
    dist = zeros( size( feature , 1 ) , 1 ) ;
    for i = 1 : size( feature , 1 )
        dist( i ) = norm( test_feature - feature( i , : ) ) ;
    end
    [ ~ , l ] = min( dist ) ;
    l = label( l ) ;
%     disp( '!' ) ;
end
disp( sprintf( '˵�Ĵ�����: %s' , label_list{ l } ) ) ;