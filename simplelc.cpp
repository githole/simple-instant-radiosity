#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <queue>
#include <vector>
#include <map>
#include <list>
#include <algorithm>

const double PI = 3.14159265358979323846;
const double INF = 1e20;
const double EPS = 1e-6;
const double MaxDepth = 5;

// *** その他の関数 ***
inline double clamp(double x){ return x<0 ? 0 : x>1 ? 1 : x; } 
inline int toInt(double x){ return int(pow(clamp(x),1/2.2)*255+.5); } 
inline double rand01() { return (double)rand()/RAND_MAX; }

// *** データ構造 ***
struct Vec {
	double x, y, z;
	Vec(const double x_ = 0, const double y_ = 0, const double z_ = 0) : x(x_), y(y_), z(z_) {}
	inline Vec operator+(const Vec &b) const {return Vec(x + b.x, y + b.y, z + b.z);}
	inline Vec operator-(const Vec &b) const {return Vec(x - b.x, y - b.y, z - b.z);}
	inline Vec operator*(const double b) const {return Vec(x * b, y * b, z * b);}
	inline Vec operator/(const double b) const {return Vec(x / b, y / b, z / b);}
	inline const double LengthSquared() const { return x*x + y*y + z*z; }
	inline const double Length() const { return sqrt(LengthSquared()); }
};
inline Vec operator*(double f, const Vec &v) { return v * f; }
inline Vec Normalize(const Vec &v) { return v / v.Length(); }
// 要素ごとの積をとる
inline const Vec Multiply(const Vec &v1, const Vec &v2) {
	return Vec(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}
inline const double Dot(const Vec &v1, const Vec &v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
inline const Vec Cross(const Vec &v1, const Vec &v2) {
	return Vec((v1.y * v2.z) - (v1.z * v2.y), (v1.z * v2.x) - (v1.x * v2.z), (v1.x * v2.y) - (v1.y * v2.x));
}
typedef Vec Color;
const Color BackgroundColor(0.0, 0.0, 0.0);

struct Ray {
	Vec org, dir;
	Ray(const Vec org_, const Vec &dir_) : org(org_), dir(dir_) {}
};

enum ReflectionType {
	DIFFUSE,    // 完全拡散面。いわゆるLambertian面。
	SPECULAR,   // 理想的な鏡面。
	REFRACTION, // 理想的なガラス的物質。
};

struct Sphere {
	double radius;
	Vec position;
	Color emission, color;
	ReflectionType ref_type;

	Sphere(const double radius_, const Vec &position_, const Color &emission_, const Color &color_, const ReflectionType ref_type_) :
	  radius(radius_), position(position_), emission(emission_), color(color_), ref_type(ref_type_) {}
	// 入力のrayに対する交差点までの距離を返す。交差しなかったら0を返す。
	const double intersect(const Ray &ray) {
		Vec o_p = position - ray.org;
		const double b = Dot(o_p, ray.dir), det = b * b - Dot(o_p, o_p) + radius * radius;
		if (det >= 0.0) {
			const double sqrt_det = sqrt(det);
			const double t1 = b - sqrt_det, t2 = b + sqrt_det;
			if (t1 > EPS)		return t1;
			else if(t2 > EPS)	return t2;
		}
		return 0.0;
	}
};

// *** レンダリングするシーンデータ ****
// from smallpt
Sphere spheres[] = {
//	Sphere(5.0, Vec(50.0, 50.0, 81.6),Color(12,12,12), Color(), DIFFUSE),//照明
	Sphere(5.0, Vec(50.0, 75.0, 81.6),Color(12,12,12), Color(), DIFFUSE),//照明
	Sphere(1e5, Vec( 1e5+1,40.8,81.6), Color(), Color(0.75, 0.25, 0.25),DIFFUSE),// 左
	Sphere(1e5, Vec(-1e5+99,40.8,81.6),Color(), Color(0.25, 0.25, 0.75),DIFFUSE),// 右
	Sphere(1e5, Vec(50,40.8, 1e5),     Color(), Color(0.75, 0.75, 0.75),DIFFUSE),// 奥
	Sphere(1e5, Vec(50,40.8,-1e5+170), Color(), Color(), DIFFUSE),// 手前
	Sphere(1e5, Vec(50, 1e5, 81.6),    Color(), Color(0.75, 0.75, 0.75),DIFFUSE),// 床
	Sphere(1e5, Vec(50,-1e5+81.6,81.6),Color(), Color(0.75, 0.75, 0.75),DIFFUSE),// 天井
	Sphere(16.5,Vec(27,16.5,47),       Color(), Color(0.75, 0.75, 0.75), DIFFUSE),// ボール
	Sphere(16.5,Vec(73,16.5,78),       Color(), Color(0.75, 0.75, 0.75), DIFFUSE),// ボール
};
const int LightID = 0;

// *** レンダリング用関数 ***
// シーンとの交差判定関数
inline bool intersect_scene(const Ray &ray, double *t, int *id) {
	const double n = sizeof(spheres) / sizeof(Sphere);
	*t  = INF;
	*id = -1;
	for (int i = 0; i < int(n); i ++) {

		double d = spheres[i].intersect(ray);
		if (d > 0.0 && d < *t) {
			*t  = d;
			*id = i;
		}
	}
	return *t < INF;
}

struct PointLight {
	Vec position;
	Color power;
	Vec normal;

	PointLight(const Vec& position_, const Vec& normal_, const Color& power_) :
	position(position_), normal(normal_), power(power_) {}
};

void emit_vpl(const int vpl_num, std::vector<PointLight> *point_lights) {
	for (int i = 0; i < vpl_num; i ++) {
		// 光源からフォトンを発射する
		// 光源上の一点をサンプリングする	

		const double r1 = 2 * PI * rand01();
		const double r2 = 1.0 - 2.0 * rand01() ;

		const Vec light_pos = spheres[LightID].position + ((spheres[LightID].radius + EPS) * Vec(sqrt(1.0 - r2*r2) * cos(r1), sqrt(1.0 - r2*r2) * sin(r1), r2));

		const Vec normal = Normalize(light_pos - spheres[LightID].position);
		// 光源上の点から半球サンプリングする
		Vec w, u, v;
		w = normal;
		if (fabs(w.x) > 0.1)
			u = Normalize(Cross(Vec(0.0, 1.0, 0.0), w));
		else
			u = Normalize(Cross(Vec(1.0, 0.0, 0.0), w));
		v = Cross(w, u);
		// コサイン項に比例させる。フォトンが運ぶのが放射輝度ではなく放射束であるため。
		const double u1 = 2 * PI * rand01();
		const double u2 = rand01(), u2s = sqrt(u2);
		Vec light_dir = Normalize((u * cos(u1) * u2s + v * sin(u1) * u2s + w * sqrt(1.0 - u2)));

		Ray now_ray(light_pos, light_dir);
		// emissionの値は放射輝度だが、フォトンが運ぶのは放射束なので変換する必要がある。
		// L（放射輝度）= dΦ/(cosθdωdA)なので、光源の放射束はΦ = ∫∫L・cosθdωdAになる。今回は球光源で完全拡散光源であることから
		// 球上の任意の場所、任意の方向に等しい放射輝度Leを持つ。（これがemissionの値）よって、
		// Φ = Le・∫∫cosθdωdAで、Le・∫dA∫cosθdωとなり、∫dAは球の面積なので4πr^2、∫cosθdωは立体角の積分なのでπとなる。
		// よって、Φ = Le・4πr^2・πとなる。この値を光源から発射するフォトン数で割ってやれば一つのフォトンが運ぶ放射束が求まる。
		Color now_flux = spheres[LightID].emission * 4.0 * PI * pow(spheres[LightID].radius, 2.0) * PI / vpl_num;

		point_lights->push_back(PointLight(light_pos, normal, now_flux));

		// フォトンがシーンを飛ぶ
		bool trace_end = false;
		for (;!trace_end;) {
			// 放射束が0.0なフォトンを追跡してもしょうがないので打ち切る
			if (std::max(now_flux.x, std::max(now_flux.y, now_flux.z)) <= 0.0)
				break;

			double t; // レイからシーンの交差位置までの距離
			int id;   // 交差したシーン内オブジェクトのID
			if (!intersect_scene(now_ray, &t, &id))
				break;
			const Sphere &obj = spheres[id];
			const Vec hitpoint = now_ray.org + t * now_ray.dir; // 交差位置
			const Vec normal  = Normalize(hitpoint - obj.position); // 交差位置の法線
			const Vec orienting_normal = Dot(normal, now_ray.dir) < 0.0 ? normal : (-1.0 * normal); // 交差位置の法線（物体からのレイの入出を考慮）

			switch (obj.ref_type) {
			case DIFFUSE: {
				// 拡散面なのでフォトンをフォトンマップに格納する
				point_lights->push_back(PointLight(hitpoint, orienting_normal, now_flux));

				// 反射するかどうかをロシアンルーレットで決める
				// 例によって確率は任意。今回はフォトンマップ本に従ってRGBの反射率の平均を使う
				const double probability = (obj.color.x + obj.color.y + obj.color.z) / 3;
				if (probability > rand01()) { // 反射
					// orienting_normalの方向を基準とした正規直交基底(w, u, v)を作る。この基底に対する半球内で次のレイを飛ばす。
					Vec w, u, v;
					w = orienting_normal;
					if (fabs(w.x) > 0.1)
						u = Normalize(Cross(Vec(0.0, 1.0, 0.0), w));
					else
						u = Normalize(Cross(Vec(1.0, 0.0, 0.0), w));
					v = Cross(w, u);
					// コサイン項を使った重点的サンプリング
					const double r1 = 2 * PI * rand01();
					const double r2 = rand01(), r2s = sqrt(r2);
					Vec dir = Normalize((u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1.0 - r2)));
					
					now_ray = Ray(hitpoint, dir);
					now_flux = Multiply(now_flux, obj.color) / probability;
					
					continue;
				} else { // 吸収（すなわちここで追跡終了）
					trace_end = true;
					continue;
				}
			} break;
			case SPECULAR: {
				// 完全鏡面なのでフォトン格納しない
				// 完全鏡面なのでレイの反射方向は決定的。
				now_ray = Ray(hitpoint, now_ray.dir - normal * 2.0 * Dot(normal, now_ray.dir));
				now_flux = Multiply(now_flux, obj.color);
				continue;
			} break;
			case REFRACTION: {
				// やはりフォトン格納しない
				Ray reflection_ray = Ray(hitpoint, now_ray.dir - normal * 2.0 * Dot(normal, now_ray.dir));
				bool into = Dot(normal, orienting_normal) > 0.0; // レイがオブジェクトから出るのか、入るのか

				// Snellの法則
				const double nc = 1.0; // 真空の屈折率
				const double nt = 1.5; // オブジェクトの屈折率
				const double nnt = into ? nc / nt : nt / nc;
				const double ddn = Dot(now_ray.dir, orienting_normal);
				const double cos2t = 1.0 - nnt * nnt * (1.0 - ddn * ddn);
		
				if (cos2t < 0.0) { // 全反射した
					now_ray = reflection_ray;
					now_flux = Multiply(now_flux, obj.color);
					continue;
				}
				// 屈折していく方向
				Vec tdir = Normalize(now_ray.dir * nnt - normal * (into ? 1.0 : -1.0) * (ddn * nnt + sqrt(cos2t)));
				const double probability  = 0.5;

				// 屈折と反射のどちらか一方を追跡する。
				// ロシアンルーレットで決定する。
				if (rand01() < probability) { // 反射
					now_ray = Ray(hitpoint, tdir);
					now_flux = Multiply(now_flux, obj.color);
					continue;
				} else { // 屈折
					now_ray = reflection_ray;
					now_flux = Multiply(now_flux, obj.color);
					continue;
				}
			} break;
			}
		}
	}
	std::cout << "VPL: " << point_lights->size() << std::endl;
}

inline double luminance(const Color &color) {
	return Dot(Vec(0.2126, 0.7152, 0.0722), color);
}

class LightTree {
private:
	struct Cluster {
		std::vector<PointLight *> representative_lights; // 輝度計算時に使う
		std::vector<double> cdf;

		Color total_intensity;
		PointLight *representative_light; // 適当に使う
		Cluster *left;
		Cluster *right;
		
		// クラスターのAABBの二頂点
		Vec vmin, vmax;

		Cluster(const Color &total_intensity_, PointLight *representative_light_) : total_intensity(total_intensity_), representative_light(representative_light_) {
			left = right = NULL;
			vmin = vmax = representative_light->position;
			representative_lights.push_back(representative_light);
		}

		// クラスタの代表点光源を取得する。
		// クラスタ生成の時に使った代表点光源とは別に、点光源候補の中から確率的に選ぶ。
		// それによってバンドアーティファクト軽減する。（そのかわりノイズが発生する）
		PointLight* get_representative_light() const {
			double d = rand01();
			std::vector<double>::const_iterator ite = std::lower_bound(cdf.begin(), cdf.end(), d);
			const int num = ite - cdf.begin();
			return representative_lights[num];
		}

		// 二つのクラスタから新しいクラスタ作る
		Cluster(Cluster *c0, Cluster *c1) {
			total_intensity = c0->total_intensity + c1->total_intensity;
			
			// 新しいクラスタの代表点光源候補の個数を決定
			int next_r_lights;
			const int m0 = std::max(c0->representative_lights.size(), c1->representative_lights.size());
			const int m1 = std::min(c0->representative_lights.size(), c1->representative_lights.size());

			const double prob = (double) m1 / m0;
			if (prob > rand01()) {
				next_r_lights = m0 + 1;
			} else {
				next_r_lights = m0;
			}

			// 新しいクラスタの代表点光源候補を決定
			double accum = 0.0;
			std::vector<PointLight *> tmp;
			tmp.reserve(c0->representative_lights.size() + c1->representative_lights.size());
			int ia = 0, ib = 0;
			for (;;) {
				if (ia < c0->representative_lights.size() && ib < c1->representative_lights.size()) {
					if (luminance(c0->representative_lights[ia]->power) < luminance(c1->representative_lights[ib]->position)) {
						accum += luminance(c0->representative_lights[ia]->power);
						tmp.push_back(c0->representative_lights[ia]);
						ia ++;
					} else {
						accum += luminance(c1->representative_lights[ib]->power);
						tmp.push_back(c1->representative_lights[ib]);
						ib ++;
					}
				} else if (ia < c0->representative_lights.size()) {
					accum += luminance(c0->representative_lights[ia]->power);
					tmp.push_back(c0->representative_lights[ia]);
					ia ++;
				} else if (ib < c1->representative_lights.size()) {
					accum += luminance(c1->representative_lights[ib]->power);
					tmp.push_back(c1->representative_lights[ib]);
					ib ++;
				} else break;

			}

			// 輝度の大きい順に候補にしていく
			// ついでに、後で確率的に選ぶために使う累積分布関数も作る
			accum = 0.0;
			for (int i = 0; i < next_r_lights; i ++) {
				double d = rand01() * accum;
				representative_lights.push_back(tmp[tmp.size() - i - 1]);
				accum += luminance(tmp[tmp.size() - i - 1]->power);
				cdf.push_back(accum);
			}
			for (int i = 0; i < cdf.size(); i ++)
				cdf[i] /= accum;
			
			// 代表点光源を確率的に決める。これはクラスタリングに使う。
			const double probability = luminance(c0->total_intensity) / (luminance(c0->total_intensity) + luminance(c1->total_intensity));
			if (probability > rand01()) {
				representative_light = c0->representative_light;
			} else {
				representative_light = c1->representative_light;
			}

			left = c0;
			right = c1;
			
			// 二つのクラスタを包む新しいAABBを作る
			vmin = Vec(std::min(c0->vmin.x, c1->vmin.x), std::min(c0->vmin.y, c1->vmin.y), std::min(c0->vmin.z, c1->vmin.z));
			vmax = Vec(std::max(c0->vmax.x, c1->vmax.x), std::max(c0->vmax.y, c1->vmax.y), std::max(c0->vmax.z, c1->vmax.z));
		}
	};

	// 実際に計算するときのデータ格納用のデータ構造
	struct ClusterTmp {
	public:
		const Cluster *cluster;
		const PointLight* representative_light;
		double error;
		Color effect;

		bool operator<(const ClusterTmp& b) const {
			return error < b.error;
		}

		ClusterTmp(const Cluster *cluster_, const Vec &x, const Vec &dir, const Vec &normal, const Sphere &obj) :
		cluster(cluster_), representative_light(cluster_->get_representative_light()) {
			effect = calc_effect(x, dir, normal, obj);

			if (cluster->left == NULL && cluster->right == NULL) {
				error = 0;
				return;
			}

			Color accum;
			if (cluster->left != NULL)
				accum = accum + Multiply(cluster->left->total_intensity, calc_without_intensity(x, dir, normal, obj, cluster->left));
	
			if (cluster->right != NULL) 
				accum = accum + Multiply(cluster->right->total_intensity, calc_without_intensity(x, dir, normal, obj, cluster->right));
	
			error = fabs(luminance(Multiply(cluster->total_intensity, calc_without_intensity(x, dir, normal, obj, cluster)) - accum));
		}

		// 大ざっぱにクラスターからの影響を計算する
		// エラーメトリクス用
		inline Color calc_without_intensity(const Vec &x, const Vec &dir, const Vec &normal, const Sphere &obj, const Cluster *now_root) {
			PointLight* pl = now_root->get_representative_light();

			const Vec to0 = Normalize(x - pl->position);
			const Vec to1 = Normalize(pl->position - x);
			const double dist2 = (x - pl->position).LengthSquared();
		
			return (1 / (dist2 + 256)) * obj.color / PI / PI;
		}

		// 真面目に影響を計算する
		inline Color calc_effect(const Vec &x, const Vec &dir, const Vec &normal, const Sphere &obj) {
			// 現在のクラスタからの影響計算
			int id;
			double t;
			// v0 <-> v1 の可視判定
			const PointLight* pl = representative_light;
			const double dist = (x - pl->position).LengthSquared();
			const Vec to0 = Normalize(x - pl->position);
			const Vec to1 = Normalize(pl->position - x);
			intersect_scene(Ray(pl->position, to0), &t, &id);
			const double c0 = Dot(pl->normal, to0);
			const double c1 = Dot(normal, to1);
			const double dist2 = (x - pl->position).LengthSquared();

			if (c0 >= 0 &&  c1 >= 0 && fabs(sqrt(dist2) - t) < EPS) {
				return (c0 * c1 / (dist2 + 256)) * Multiply(obj.color / PI, cluster->total_intensity) / PI;
			}
			return Color();
		}
	};

	void delete_light_tree(Cluster *now_root) {
		if (now_root == NULL)
			return;

		delete_light_tree(now_root->left);
		delete_light_tree(now_root->right);

		delete now_root;
	}

	int MaxCluster;
	int MinCluster;
public:
	int TotalCutSize;
	Cluster *root;

	~LightTree() {
		delete_light_tree(root);
	}

	// クラスタから放射輝度計算
	Color radiance(const Vec &x, const Vec &dir, const Vec &normal, const Sphere &obj) {
		MaxCluster = 1024;
		MinCluster = 16;
		
		ClusterTmp root_tmp = ClusterTmp(root, x, dir, normal, obj);
		Color accum = root_tmp.effect;

		std::priority_queue<ClusterTmp> now_clusters;
		now_clusters.push(root_tmp);

		// エラーの順に並んだプライオリティキュー使う
		for (;now_clusters.size() < MaxCluster;) {
			ClusterTmp tmp = now_clusters.top();

			// 累計放射輝度に対して一定割合以上に誤差があるクラスタはどんどん分割していく
			if (tmp.error > 0.001 * luminance(accum)) {
				now_clusters.pop();
				accum = accum - tmp.effect;

				Cluster* left = tmp.cluster->left;
				Cluster* right = tmp.cluster->right;

				if (left != NULL) {
					ClusterTmp left_tmp = ClusterTmp(left, x, dir, normal, obj);
					now_clusters.push(left_tmp);
					accum = accum + left_tmp.effect;
				}
				if (right != NULL) {
					ClusterTmp right_tmp = ClusterTmp(right, x, dir, normal, obj);
					now_clusters.push(right_tmp);
					accum = accum + right_tmp.effect;
				}

			} else {
				break;
			}
		}
		TotalCutSize += now_clusters.size();
		return accum;
	}

	static double size_metric(const Cluster &c0, const Cluster &c1) {
		// 二つのクラスタを包む新しいAABBを作る
		const Vec nmin = Vec(std::min(c0.vmin.x, c1.vmin.x), std::min(c0.vmin.y, c1.vmin.y), std::min(c0.vmin.z, c1.vmin.z));
		const Vec nmax = Vec(std::max(c0.vmax.x, c1.vmax.x), std::max(c0.vmax.y, c1.vmax.y), std::max(c0.vmax.z, c1.vmax.z));

		const double a = (nmax - nmin).Length();
		const double I = luminance(c0.total_intensity + c1.total_intensity);
		return a * a * I;
	}

	struct ClusterPair {
		double metric;
		Cluster *pair0, *pair1;

		ClusterPair() {}

		ClusterPair(Cluster *pair0_, Cluster *pair1_) : pair0(pair0_), pair1(pair1_) {
			metric = size_metric(*pair0, *pair1);
		}
		
		bool operator>(const ClusterPair& b) const {
			return metric > b.metric;
		}
	};

	void CreateLightTree(std::vector<PointLight> &point_lights) {
		TotalCutSize = 0;

		// 最初に葉を作る
		std::vector<Cluster*> leafs;
		for (int i = 0; i < point_lights.size(); i ++) {
			leafs.push_back(new Cluster(point_lights[i].power, &point_lights[i]));
		}

		// ボトムアップにLightTreeを作る
		// 乱数つかって適当に作る
		// 本当は真面目に最適化しないとしょぼいクラスターになって終わる
		// O(nlogn)のアルゴリズムあるのでそのうち作る（かもしれない）
		for (;;) {
			//std::cout << "*";
			std::vector<Cluster *> new_clusters(leafs.size() - 1);

			int pair0 = -1, pair1 = -1;
			double now_cluster_size = INF;
			int idx = rand() % leafs.size();
			for (int j = 0; j < leafs.size(); j ++) {
				if (idx == j)
					continue;
				double size = size_metric(*leafs[idx], *leafs[j]);
				if (size < now_cluster_size) {
					now_cluster_size = size;
					pair0 = idx;
					pair1 = j;
				}
			}
			Cluster *cluster = new Cluster(leafs[pair0], leafs[pair1]);

			idx = 0;
			for (int i = 0; i < leafs.size(); i ++) {
				if (i != pair0 && i != pair1) {
					new_clusters[idx] = leafs[i];
					idx ++;
				}
			}
			new_clusters[idx] = cluster;

			leafs.swap(new_clusters);

			if (leafs.size() == 1) {	
				root = leafs[0];
				break;
			}
		}
	}
};


// ray方向からの放射輝度を求める
Color radiance(const Ray &ray, const int depth, LightTree *light_tree) {
	double t; // レイからシーンの交差位置までの距離
	int id;   // 交差したシーン内オブジェクトのID
	if (!intersect_scene(ray, &t, &id))
		return BackgroundColor;

	const Sphere &obj = spheres[id];
	const Vec hitpoint = ray.org + t * ray.dir; // 交差位置
	const Vec normal  = Normalize(hitpoint - obj.position); // 交差位置の法線
	const Vec orienting_normal = Dot(normal, ray.dir) < 0.0 ? normal : (-1.0 * normal); // 交差位置の法線（物体からのレイの入出を考慮）

	// 色の反射率最大のものを得る。ロシアンルーレットで使う。
	// ロシアンルーレットの閾値は任意だが色の反射率等を使うとより良い。
	double russian_roulette_probability = std::max(obj.color.x, std::max(obj.color.y, obj.color.z));
	// 一定以上レイを追跡したらロシアンルーレットを実行し追跡を打ち切るかどうかを判断する
	if (depth > MaxDepth) {
		if (rand01() >= russian_roulette_probability)
			return obj.emission;
	} else
		russian_roulette_probability = 1.0; // ロシアンルーレット実行しなかった
		
	switch (obj.ref_type) {
	case DIFFUSE: {
		if (id == LightID)
			return obj.emission;

		return light_tree->radiance(hitpoint, ray.dir, orienting_normal, obj);
	} break;


	// SPECULARとREFRACTIONの場合はパストレーシングとほとんど変わらない。
	// 単純に反射方向や屈折方向の放射輝度(Radiance)をradiance()で求めるだけ。

	case SPECULAR: {
		// 完全鏡面にヒットした場合、反射方向から放射輝度をもらってくる
		return obj.emission + radiance(Ray(hitpoint, ray.dir - normal * 2.0 * Dot(normal, ray.dir)), depth+1, light_tree);
	} break;
	case REFRACTION: {
		Ray reflection_ray = Ray(hitpoint, ray.dir - normal * 2.0 * Dot(normal, ray.dir));
		bool into = Dot(normal, orienting_normal) > 0.0; // レイがオブジェクトから出るのか、入るのか

		// Snellの法則
		const double nc = 1.0; // 真空の屈折率
		const double nt = 1.5; // オブジェクトの屈折率
		const double nnt = into ? nc / nt : nt / nc;
		const double ddn = Dot(ray.dir, orienting_normal);
		const double cos2t = 1.0 - nnt * nnt * (1.0 - ddn * ddn);

		if (cos2t < 0.0) { // 全反射した	
			// 反射方向から放射輝度をもらってくる
			return obj.emission + Multiply(obj.color,
				radiance(Ray(hitpoint, ray.dir - normal * 2.0 * Dot(normal, ray.dir)), depth+1, light_tree)) / russian_roulette_probability;
		}
		// 屈折していく方向
		Vec tdir = Normalize(ray.dir * nnt - normal * (into ? 1.0 : -1.0) * (ddn * nnt + sqrt(cos2t)));

		// SchlickによるFresnelの反射係数の近似
		const double a = nt - nc, b = nt + nc;
		const double R0 = (a * a) / (b * b);
		const double c = 1.0 - (into ? -ddn : Dot(tdir, normal));
		const double Re = R0 + (1.0 - R0) * pow(c, 5.0);
		const double Tr = 1.0 - Re; // 屈折光の運ぶ光の量
		const double probability  = 0.25 + 0.5 * Re;

		// 一定以上レイを追跡したら屈折と反射のどちらか一方を追跡する。（さもないと指数的にレイが増える）
		// ロシアンルーレットで決定する。
		if (depth > 2) {
			if (rand01() < probability) { // 反射
				return obj.emission +
					Multiply(obj.color, radiance(reflection_ray, depth+1, light_tree) * Re)
					/ probability
					/ russian_roulette_probability;
			} else { // 屈折
				return obj.emission +
					Multiply(obj.color, radiance(Ray(hitpoint, tdir), depth+1, light_tree) * Tr)
					/ (1.0 - probability)
					/ russian_roulette_probability;
			}
		} else { // 屈折と反射の両方を追跡
			return obj.emission +
				Multiply(obj.color, radiance(reflection_ray, depth+1, light_tree) * Re
				+ radiance(Ray(hitpoint, tdir), depth+1, light_tree) * Tr) / russian_roulette_probability;
		}
	} break;
	}

	return Color();
}


// *** .hdrフォーマットで出力するための関数 ***
struct HDRPixel {
	unsigned char r, g, b, e;
	HDRPixel(const unsigned char r_ = 0, const unsigned char g_ = 0, const unsigned char b_ = 0, const unsigned char e_ = 0) :
	r(r_), g(g_), b(b_), e(e_) {};
	unsigned char get(int idx) {
		switch (idx) {
		case 0: return r;
		case 1: return g;
		case 2: return b;
		case 3: return e;
		} return 0;
	}

};

// doubleのRGB要素を.hdrフォーマット用に変換
HDRPixel get_hdr_pixel(const Color &color) {
	double d = std::max(color.x, std::max(color.y, color.z));
	if (d <= 1e-32)
		return HDRPixel();
	int e;
	double m = frexp(d, &e); // d = m * 2^e
	d = m * 256.0 / d;
	return HDRPixel(color.x * d, color.y * d, color.z * d, e + 128);
}

// 書き出し用関数
void save_hdr_file(const std::string &filename, const Color* image, const int width, const int height) {
	FILE *fp = fopen(filename.c_str(), "wb");
	if (fp == NULL) {
		std::cerr << "Error: " << filename << std::endl;
		return;
	}
	// .hdrフォーマットに従ってデータを書きだす
	// ヘッダ
	unsigned char ret = 0x0a;
	fprintf(fp, "#?RADIANCE%c", (unsigned char)ret);
	fprintf(fp, "# Made with 100%% pure HDR Shop%c", ret);
	fprintf(fp, "FORMAT=32-bit_rle_rgbe%c", ret);	
	fprintf(fp, "EXPOSURE=1.0000000000000%c%c", ret, ret);

	// 輝度値書き出し
	fprintf(fp, "-Y %d +X %d%c", height, width, ret);
	for (int i = height - 1; i >= 0; i --) {
		std::vector<HDRPixel> line;
		for (int j = 0; j < width; j ++) {
			HDRPixel p = get_hdr_pixel(image[j + i * width]);
			line.push_back(p);
		}
		fprintf(fp, "%c%c", 0x02, 0x02);
		fprintf(fp, "%c%c", (width >> 8) & 0xFF, width & 0xFF);
		for (int i = 0; i < 4; i ++) {
			for (int cursor = 0; cursor < width;) {
				const int cursor_move = std::min(127, width - cursor);
				fprintf(fp, "%c", cursor_move);
				for (int j = cursor;  j < cursor + cursor_move; j ++)
					fprintf(fp, "%c", line[j].get(i));
				cursor += cursor_move;
			}
		}
	}

	fclose(fp);
}

int main(int argc, char **argv) {
	int width = 640;
	int height = 480;
	int samples = 1;
	int vpl = 1000;

	// カメラ位置
	Ray camera(Vec(50.0, 52.0, 295.6), Normalize(Vec(0.0, -0.042612, -1.0)));
	// シーン内でのスクリーンのx,y方向のベクトル
	Vec cx = Vec(width * 0.5135 / height);
	Vec cy = Normalize(Cross(cx, camera.dir)) * 0.5135;
	Color *image = new Color[width * height];
		
	std::vector<PointLight> point_lights;
	emit_vpl(vpl, &point_lights);

	LightTree light_tree;
	light_tree.CreateLightTree(point_lights);

	std::cout << light_tree.root->representative_lights.size() << " ";
// #pragma omp parallel for schedule(dynamic, 1)
	for (int y = 0; y < height; y ++) {
		std::cerr << "Rendering (" << samples * 4 << " spp) " << (100.0 * y / (height - 1)) << "%" << std::endl;
		srand(y * y * y);
		for (int x = 0; x < width; x ++) {
			int image_index = y * width + x;	
			image[image_index] = Color();

			// 2x2のサブピクセルサンプリング
			for (int sy = 0; sy < 2; sy ++) {
				for (int sx = 0; sx < 2; sx ++) {
					Color accumulated_radiance = Color();
					// テントフィルターによってサンプリング
					// ピクセル範囲で一様にサンプリングするのではなく、ピクセル中央付近にサンプルがたくさん集まるように偏りを生じさせる
					const double r1 = 2.0 * rand01(), dx = r1 < 1.0 ? sqrt(r1) - 1.0 : 1.0 - sqrt(2.0 - r1);
					const double r2 = 2.0 * rand01(), dy = r2 < 1.0 ? sqrt(r2) - 1.0 : 1.0 - sqrt(2.0 - r2);
					Vec dir = cx * (((sx + 0.5 + dx) / 2.0 + x) / width - 0.5) +
								cy * (((sy + 0.5 + dy) / 2.0 + y) / height- 0.5) + camera.dir;
					accumulated_radiance = accumulated_radiance + 
						radiance(Ray(camera.org + dir * 130.0, Normalize(dir)), 0, &light_tree);
					image[image_index] = image[image_index] + accumulated_radiance;
				}
			}
		}
	}
	std::cout << (double)light_tree.TotalCutSize / (width * height * 4) << std::endl;
	
	// .hdrフォーマットで出力
	save_hdr_file(std::string("image.hdr"), image, width, height);
}
