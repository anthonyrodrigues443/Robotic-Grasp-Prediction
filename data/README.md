# Cornell Grasping Dataset

**Source:** Jiang, Moseson, Saxena. *Efficient Grasping from RGBD Images: Learning using a new Rectangle Representation.* ICRA 2011.
**Original URL (dead as of 2026):** `http://pr.cs.cornell.edu/grasping/rect_data/data.php`
**Working mirror:** Wayback Machine snapshot `2016-04-14`, e.g. `https://web.archive.org/web/20160414082648/http://pr.cs.cornell.edu/grasping/rect_data/temp/data01.tar.gz`

**Contents:** 885 RGB-D images of 240 distinct household objects, with ~8 019 oriented grasp rectangles
(positive and negative). Each example is one `.pcd` file (Point Cloud Data with embedded RGB),
plus `pcdNNNNcpos.txt` (positive grasp rectangles, 4 corner points each) and
`pcdNNNNcneg.txt` (negative grasp rectangles).

The dataset ships in 10 tar.gz archives (~480 MB each, ~4.8 GB total). Phase 1 uses archives
01–03 (≈ 250 images) for fast iteration; later phases will scale to the full set.

**License:** Research / academic use only (Cornell terms). Not redistributed here — download from the
Wayback URL above. `data/raw/` is gitignored.
