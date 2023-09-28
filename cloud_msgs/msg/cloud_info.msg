Header header 

int32[] ring_index_start
int32[] ring_index_end

float32 orientation_start
float32 orientation_end
float32 orientation_diff

bool[]    ground_segment_flag # true - ground point, false - other points
uint32[]  ground_segment_cloud_column # point column index in range image
float32[] ground_segment_cloud_range # point range 