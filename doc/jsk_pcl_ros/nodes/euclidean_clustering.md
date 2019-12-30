# EuclideanClustering
![](images/euclidean_segmentation.png)
## What Is This
Segment pointcloud based euclidean metrics, which is based on `pcl::EuclideanClusterExtraction`.
This nodelet has topic interface and service interface.

The result of clustering is published as `jsk_recognition_msgs/ClusterPointIndices`.

If the number of the cluster is not changed across different frames, `EuclideanClustering`
tries to track the segment.

## Subscribing Topics
* `~input` (`sensor_msgs/PointCloud2`):

   input pointcloud. If `multi` is `talse`, this input is only enough.


* `~input/indices` (`jsk_recognition_msgs/ClusterPointIndices`):

   input indices. If `multi` is `true`, synchronized `~input` and `~input/indices` are used.

## Publishing Topics
* `~output` (`jsk_recognition_msgs/ClusterPointIndices`):

   Result of clustering.
* `~cluster_num` (`jsk_recognition_msgs/Int32Stamped`):

   The number of clusters.

## Advertising Services
* `~euclidean_clustering` (`jsk_pcl_ros/EuclideanSegment`):

   Service interface to segment clusters.

```
sensor_msgs/PointCloud2 input
float32 tolerance
---
sensor_msgs/PointCloud2[] output
```

## Parameters
* `~tolerance` (Double, default: `0.02`):

   Max distance for the points to be regarded as same cluster.
* `~label_tracking_tolerance` (Double, default: `0.2`)

   Max distance to track the cluster between different frames.
* `~max_size` (Integer, default: `25000`)

   The maximum number of the points of one cluster.
* `~min_size` (Integer, default: `20`)

   The minimum number of the points of one cluster.

* `~multi` (Boolean, default: `false`)

   Flag of multi euclidean clustering. If `~multi` is `true`, synchronized `~input` and `~input/indices` are used.

* `~approximate_sync` (Boolean, default: `False`):

   Policy of synchronization, if `false` it synchornizes exactly, else approximately.
   This value is only valid in case of `~multi` is `true`.

* `~queue_size` (Int, default: `1`):

   Queue size of topic msgs.

* `~downsample_enable` (Boolean, default: `false`)

   Flag of VoxelGrid downsampling. If `~downsample_enable` is `true`, `~input` is downsampled.

* `~leaf_size` (Double, default: `0.01`)

   Leaf size of voxel grid downsampling.
   This value is only valid in case of `~downsample_enable` is `true`.


## Sample
Plug the depth sensor which can be launched by openni.launch and run the below command.

```
roslaunch jsk_pcl_ros euclidean_segmentation.launch
```
