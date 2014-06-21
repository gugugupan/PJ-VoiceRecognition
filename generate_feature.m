% Train Set 001 002 004 005 006 009
% Test Set 003 007 008 010
% File  000 - 023 for different label

% extract_set = [ 1 , 2 , 4 , 5 , 6 , 9 ] ;
extract_set = [ 3 , 7 , 8 , 10 ] ;
train_file_count = 24 ;

all_feature = zeros( 0 , 24 ) ;
mfcc_feature = cell( length( extract_set ) * train_file_count , 1 ) ;
label = zeros( length( extract_set ) * train_file_count , 1 ) ;
l = 1 ;

for set_i = 1 : length( extract_set ) 
    set_id = extract_set( set_i ) ;
    for i = 0 : train_file_count - 1 
        mfcc_feature{ l } = mfcc( sprintf( './data/%03d/words/%03d.wav' , set_id , i ) ) ;
        label( l ) = i + 1 ;
        all_feature = [ all_feature ; mfcc_feature{ l } ] ;
        l = l + 1 ;
        disp( sprintf( 'Extract MFCC Feature: %d-%d' , set_id , i ) ) ;
    end
end

disp( 'Calc Codebook' ) ;
% codebook_size = 500 ;
% [ ~ , codebook ] = kmeans( all_feature , codebook_size ) ; 
% save( 'codebook.mat' , 'codebook' , 'codebook_size' ) ;

feature = zeros( length( mfcc_feature ) , 500 ) ;
dist = zeros( code_book_size , 1 ) ;
for i = 1 : length( mfcc_feature )
    for j = 1 : size( mfcc_feature{ i } , 1 )
        for k = 1 : code_book_size
            dist( k ) = norm( codebook( k , : ) - mfcc_feature{ i } ( j , : ) ) ;
        end
        [ ~ , k ] = min( dist ) ;
        feature( i , k ) = feature( i , k ) + 1 ;
    end
    disp( sprintf( 'Extract Feature %d/%d' , i , length( mfcc_feature ) ) ) ;
end

save( 'feature.mat' , 'feature' , 'label' ) ;
