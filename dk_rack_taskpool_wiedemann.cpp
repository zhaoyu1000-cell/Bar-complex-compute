#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparse_rank_wiedemann_parallel.cpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using std::int64_t;
// (same utilities as provided)
static inline int64_t mod_pow(int64_t a, int64_t e, int64_t p){int64_t r=1%p; a%=p; while(e>0){if(e&1) r=(int64_t)((__int128)r*a%p); a=(int64_t)((__int128)a*a%p); e>>=1;} return r;}
static inline int64_t mod_inv(int64_t a, int64_t p){a%=p; if(a<0)a+=p; return mod_pow(a,p-2,p);} 
static bool is_prime_ll(int64_t n){ if(n<2) return false; if((n%2)==0) return n==2; for(int64_t d=3; d*d<=n; d+=2) if(n%d==0) return false; return true; }
static std::vector<int64_t> factor_distinct(int64_t n){ std::vector<int64_t> fac; for(int64_t d=2; d*d<=n; d+=(d==2?1:2)){ if(n%d==0){fac.push_back(d); while(n%d==0) n/=d;}} if(n>1) fac.push_back(n); return fac; }
static int64_t primitive_root_mod_prime(int64_t p){ const int64_t phi=p-1; auto fac=factor_distinct(phi); for(int64_t g=2; g<p; g++){ bool ok=true; for(auto q:fac){ if(mod_pow(g,phi/q,p)==1){ok=false; break;}} if(ok) return g;} throw std::runtime_error("No primitive root found"); }
static int64_t pick_prime_1_mod_m(int64_t m,int64_t start=1000000){ if(m==1) return 1000003; int64_t t=(start-1)/m+1; while(true){int64_t cand=m*t+1; if(is_prime_ll(cand)) return cand; t++;}}
static int64_t q_scalar_in_gfp(int64_t p,int64_t m,int64_t exp,int sign){ if(sign!=1&&sign!=-1) throw std::runtime_error("sign"); if(m==1) return sign==1?1:(p-1); if(p%m!=1) throw std::runtime_error("p mod m"); int64_t g=primitive_root_mod_prime(p); int64_t z=mod_pow(g,(p-1)/m,p); int64_t q=mod_pow(z,((exp%m)+m)%m,p); if(sign==-1) q=(p-q)%p; return q; }
static __int128 binom_i128(int n,int k){ if(k<0||k>n) return 0; k=std::min(k,n-k); __int128 r=1; for(int i=1;i<=k;i++){ r=r*(n-(k-i)); r/=i;} return r; }
static void print_i128(__int128 x){ if(x==0){std::cout<<"0";return;} if(x<0){std::cout<<"-"; x=-x;} std::string s; while(x>0){ s.push_back(char('0'+(int)(x%10))); x/=10;} std::reverse(s.begin(),s.end()); std::cout<<s; }

struct Dihedral{ int N; int G; explicit Dihedral(int n):N(n),G(2*n){}; inline int gid(int a,int b) const {a%=N; if(a<0)a+=N; b&=1; return a+N*b;} inline int a_of(int g) const{return g%N;} inline int b_of(int g) const{return g/N;} inline int id() const{return gid(0,0);} inline int mul(int g1,int g2) const {int a1=a_of(g1),b1=b_of(g1),a2=a_of(g2),b2=b_of(g2); int a=(b1?(a1-a2):(a1+a2))%N; if(a<0)a+=N; return gid(a,b1^b2);} inline int inv(int g) const {int a=a_of(g),b=b_of(g); if(b==0) return gid(-a,0); return g;} inline int conj(int x,int y) const {return mul(mul(x,y),inv(x));} inline int reflection_gid(int i) const{return gid(i,1);} };

static std::vector<int64_t> powB_list(int base,int n){ std::vector<int64_t> p(n+1,1); for(int i=1;i<=n;i++) p[i]=p[i-1]*base; return p; }
static inline void decode_base(int64_t wid,int base,int L,uint8_t* out){ for(int i=0;i<L;i++){ out[i]=(uint8_t)(wid%base); wid/=base; } }
static inline int64_t encode_base(const uint8_t* d,int L,const std::vector<int64_t>& powB){ int64_t s=0; for(int i=0;i<L;i++) s += (int64_t)d[i]*powB[i]; return s; }
static inline int64_t replace_segment(int64_t wid,int start,int L,int64_t out_id,const std::vector<int64_t>& powB){ int64_t low=wid%powB[start]; int64_t high=wid/powB[start+L]; return low+out_id*powB[start]+high*powB[start+L]; }

static std::vector<int> reduced_swaps_for_perm(const std::vector<int>& w){ int m=(int)w.size(); std::vector<int> arr(m); std::iota(arr.begin(),arr.end(),0); std::vector<int> swaps; for(int i=0;i<m;i++){ if(arr[i]==w[i]) continue; int j=(int)(std::find(arr.begin(),arr.end(),w[i])-arr.begin()); for(int k=j-1;k>=i;k--){ std::swap(arr[k],arr[k+1]); swaps.push_back(k);} } return swaps; }
static std::vector<std::vector<int>> shuffle_swaplists(int p_len,int q_len){ int L=p_len+q_len; std::vector<std::vector<int>> res; std::vector<int> bit(L,0); std::fill(bit.begin(),bit.begin()+p_len,1); std::sort(bit.begin(),bit.end(),std::greater<int>()); do{ std::vector<int> perm; int iu=0,iv=0; for(int i=0;i<L;i++) perm.push_back(bit[i]?iu++:p_len+iv++); res.push_back(reduced_swaps_for_perm(perm)); }while(std::prev_permutation(bit.begin(),bit.end())); return res; }

struct QShufflePQ{ int p_len=0,q_len=0,out_len=0; std::vector<std::vector<std::pair<int64_t,int64_t>>> data; };
static inline void apply_swaps_rack(uint8_t* d,int L,const std::vector<int>& swaps,const std::vector<std::vector<uint8_t>>& op){ for(int s:swaps){ uint8_t x=d[s],y=d[s+1]; d[s]=op[x][y]; d[s+1]=x; }}

static std::vector<std::vector<QShufflePQ>> precompute_qshuffle_selected(int n,int base,int64_t P,int64_t q_scalar,const std::vector<std::vector<uint8_t>>& op,const std::vector<int64_t>& powB,const std::vector<std::pair<int,int>>& pairs){
 std::vector<int64_t> qpow(n*n+1,1); for(int i=1;i<(int)qpow.size();i++) qpow[i]=(int64_t)((__int128)qpow[i-1]*q_scalar%P);
 std::vector<std::vector<std::vector<uint8_t>>> digits(n+1); for(int L=1;L<=n;L++){ digits[L].assign((size_t)powB[L],std::vector<uint8_t>(L)); for(int64_t wid=0; wid<powB[L]; wid++) decode_base(wid,base,L,digits[L][(size_t)wid].data()); }
 std::vector<std::vector<QShufflePQ>> table(n+1,std::vector<QShufflePQ>(n+1));
 auto do_pair=[&](int p_len,int q_len){ int out_len=p_len+q_len; int64_t out_sz=powB[out_len]; QShufflePQ pq; pq.p_len=p_len; pq.q_len=q_len; pq.out_len=out_len; pq.data.assign((size_t)(powB[p_len]*powB[q_len]),{}); auto sh=shuffle_swaplists(p_len,q_len); std::array<uint8_t,64> basew{},work{}; std::vector<int64_t> acc((size_t)out_sz,0); std::vector<int64_t> touched; touched.reserve(256); for(int64_t u=0; u<powB[p_len]; u++) for(int64_t v=0; v<powB[q_len]; v++){ auto &ud=digits[p_len][(size_t)u]; auto &vd=digits[q_len][(size_t)v]; for(int i=0;i<p_len;i++) basew[i]=ud[i]; for(int j=0;j<q_len;j++) basew[p_len+j]=vd[j]; touched.clear(); for(const auto& swaps:sh){ for(int t=0;t<out_len;t++) work[t]=basew[t]; apply_swaps_rack(work.data(),out_len,swaps,op); int64_t out_id=encode_base(work.data(),out_len,powB); int64_t coeff=qpow[(int)swaps.size()]; if(acc[(size_t)out_id]==0) touched.push_back(out_id); int64_t nv=(acc[(size_t)out_id]+coeff)%P; if(nv<0) nv+=P; acc[(size_t)out_id]=nv; } int64_t idx=u*powB[q_len]+v; auto &vec=pq.data[(size_t)idx]; vec.clear(); vec.reserve(touched.size()); for(int64_t out_id:touched){ int64_t c=acc[(size_t)out_id]; if(c) vec.push_back({out_id,c}); acc[(size_t)out_id]=0; }} table[p_len][q_len]=std::move(pq); };
 if(pairs.empty()){ for(int p_len=1;p_len<=n-1;p_len++) for(int q_len=1;q_len<=n-p_len;q_len++) do_pair(p_len,q_len);} else for(auto [p,q]:pairs) do_pair(p,q);
 return table; }

static void comps_rec(int n,int k,std::vector<int>& cur,std::vector<std::vector<int>>& out){ if(k==1){cur.push_back(n); out.push_back(cur); cur.pop_back(); return;} for(int f=1;f<=n-k+1;f++){ cur.push_back(f); comps_rec(n-f,k-1,cur,out); cur.pop_back(); }}
static std::vector<std::vector<int>> compositions(int n,int k){ std::vector<std::vector<int>> out; std::vector<int> cur; comps_rec(n,k,cur,out); return out; }
static std::vector<int> cuts_for_comp(const std::vector<int>& comp){ std::vector<int> cuts; cuts.push_back(0); int s=0; for(int m:comp){s+=m; cuts.push_back(s);} return cuts; }
struct Spec{int start,p_len,q_len,out_len,new_pos; int64_t sgn;};
struct VecHash{ size_t operator()(const std::vector<int>& v) const noexcept { size_t h=0; for(int x:v) h ^= std::hash<int>{}(x + 0x9e3779b9 + (int)(h<<6)+(int)(h>>2)); return h; } };
struct KData{ int k=0; std::vector<std::vector<int>> comps_k,comps_t; std::vector<std::vector<Spec>> specs; std::vector<std::pair<int,int>> needed_pairs; };
static KData precompute_kdata(int n,int k,int64_t P){ KData kd; kd.k=k; kd.comps_k=compositions(n,k); kd.comps_t=compositions(n,k-1); std::unordered_map<std::vector<int>,int,VecHash> pos_t; for(int i=0;i<(int)kd.comps_t.size();i++) pos_t[kd.comps_t[i]]=i; kd.specs.assign(kd.comps_k.size(),{}); std::unordered_map<int,bool> seen; for(int ci=0; ci<(int)kd.comps_k.size(); ci++){ const auto& comp=kd.comps_k[ci]; auto cuts=cuts_for_comp(comp); std::vector<Spec> sp; for(int i=1;i<=k-1;i++){ int start=cuts[i-1], p_len=comp[i-1], q_len=comp[i], out_len=p_len+q_len; std::vector<int> new_comp; new_comp.insert(new_comp.end(),comp.begin(),comp.begin()+i-1); new_comp.push_back(out_len); new_comp.insert(new_comp.end(),comp.begin()+i+1,comp.end()); int new_pos=pos_t[new_comp]; int64_t sgn=((i-1)&1)?(P-1):1; sp.push_back({start,p_len,q_len,out_len,new_pos,sgn}); int key=(p_len<<8)|q_len; if(!seen[key]){seen[key]=true; kd.needed_pairs.push_back({p_len,q_len});}} kd.specs[ci]=std::move(sp);} return kd; }

struct MonoBlocks{ std::vector<std::vector<int64_t>> words_by_g; std::vector<int> idx_in_block; };
static MonoBlocks monodromy_blocks_Dn(int n,int base,const std::vector<int64_t>& powB,const Dihedral& D){ int G=D.G; int64_t Nwords=powB[n]; MonoBlocks mb; mb.words_by_g.assign(G,{}); mb.idx_in_block.assign((size_t)Nwords,-1); std::vector<int> refl(base); for(int i=0;i<base;i++) refl[i]=D.reflection_gid(i); for(int64_t wid=0; wid<Nwords; wid++){ int g=D.id(); int64_t x=wid; for(int i=0;i<n;i++){ int d=(int)(x%base); x/=base; g=D.mul(g,refl[d]); } int idx=(int)mb.words_by_g[g].size(); mb.idx_in_block[(size_t)wid]=idx; mb.words_by_g[g].push_back(wid);} return mb; }
struct ConjClasses{ std::vector<std::vector<int>> classes; std::vector<int> rep; std::vector<int> class_of; };
static ConjClasses conjugacy_classes(const Dihedral& D){ int G=D.G; ConjClasses cc; cc.class_of.assign(G,-1); for(int g=0;g<G;g++){ if(cc.class_of[g]!=-1) continue; std::vector<char> in(G,0); for(int h=0;h<G;h++) in[D.conj(h,g)] = 1; std::vector<int> cls; int rep=-1; for(int x=0;x<G;x++) if(in[x]){ cls.push_back(x); if(rep==-1||x<rep) rep=x;} int cid=(int)cc.classes.size(); cc.classes.push_back(std::move(cls)); cc.rep.push_back(rep); for(int x:cc.classes.back()) cc.class_of[x]=cid; } return cc; }

struct SparseWiedemannInput { sparse_wiedemann::SparsePairMatrix sq; };

static SparseWiedemannInput compile_block_differential_matrix(int n,int64_t P,const std::vector<std::vector<QShufflePQ>>& qsh,const std::vector<int64_t>& powB,const MonoBlocks& mb,const KData& kd,int g){
 const auto& words=mb.words_by_g[g]; int B=(int)words.size(); if(B==0) return {}; int rows=(int)kd.comps_t.size()*B; int cols=(int)kd.comps_k.size()*B; int N=std::max(rows,cols); std::vector<int64_t> acc(rows,0); std::vector<int> touched; std::vector<int> mark(rows,0); int stamp=1; auto idx_map=[&](int64_t wid){return mb.idx_in_block[(size_t)wid];}; SparseWiedemannInput mat; mat.sq.assign(N,{});
 for(int ci=0; ci<(int)kd.comps_k.size(); ci++){ const auto& sp=kd.specs[ci]; for(int c=0;c<B;c++){ int col=ci*B+c; int64_t wid=words[c]; touched.clear(); stamp++; if(stamp==0){ std::fill(mark.begin(),mark.end(),0); stamp=1; } for(const auto& ms:sp){ int64_t u=(wid/powB[ms.start])%powB[ms.p_len]; int64_t v=(wid/powB[ms.start+ms.p_len])%powB[ms.q_len]; int64_t idx=u*powB[ms.q_len]+v; const auto& lst=qsh[ms.p_len][ms.q_len].data[(size_t)idx]; int row_base=ms.new_pos*B; for(const auto& ow:lst){ int64_t out_id=ow.first, coeff=ow.second; int64_t val=(int64_t)((__int128)ms.sgn*coeff%P); if(!val) continue; int64_t new_wid=replace_segment(wid,ms.start,ms.out_len,out_id,powB); int row=row_base+idx_map(new_wid); if(mark[row]!=stamp){ mark[row]=stamp; touched.push_back(row);} int64_t nv=(acc[row]+val)%P; if(nv<0) nv+=P; acc[row]=nv; } } std::sort(touched.begin(),touched.end()); for(int r:touched){ int64_t cval=acc[r]; if(cval) mat.sq[r].push_back({col,cval}); acc[r]=0; } }}
 return mat; }

static int rank_rectangular_wiedemann(const SparseWiedemannInput& d,int64_t P,int repeats,int threads,std::uint64_t seed){ if(d.sq.empty()) return 0; return sparse_wiedemann_parallel::rank_probabilistic_parallel(d.sq,P,repeats,threads,seed); }

struct Task{int k; int rep_g; int class_size;};
int main(int argc,char** argv){
 int dihedralN=5,n=5,threads=8,only_k=-1,sign=-1,repeats=1,nested_blocks=0; int64_t m=1,exp=1,P=0;
 for(int i=1;i<argc;i++){ std::string s=argv[i]; auto need=[&](const char* opt){ if(i+1>=argc) throw std::runtime_error(std::string("Missing ")+opt); return std::string(argv[++i]);};
 if(s=="--dihedral") dihedralN=std::stoi(need("--dihedral")); else if(s=="--n") n=std::stoi(need("--n")); else if(s=="--threads") threads=std::stoi(need("--threads")); else if(s=="--only-k") only_k=std::stoi(need("--only-k")); else if(s=="--root-order") m=std::stoll(need("--root-order")); else if(s=="--root-exp") exp=std::stoll(need("--root-exp")); else if(s=="--sign") sign=std::stoi(need("--sign")); else if(s=="--prime") P=std::stoll(need("--prime")); else if(s=="--repeats") repeats=std::stoi(need("--repeats")); else if(s=="--nested-blocks") nested_blocks=std::stoi(need("--nested-blocks")); else throw std::runtime_error("Unknown"); }
 if(P==0) P=pick_prime_1_mod_m(m); int base=dihedralN; Dihedral D(dihedralN); int64_t q_scalar=q_scalar_in_gfp(P,m,exp,sign); auto powB=powB_list(base,n);
#ifdef _OPENMP
 omp_set_dynamic(0);
 if (nested_blocks) omp_set_max_active_levels(1);
#endif
 std::cout<<"D_"<<dihedralN<<" n="<<n<<" P="<<P<<" q="<<q_scalar<<" repeats="<<repeats<<"\n";
 std::vector<std::vector<uint8_t>> op(base,std::vector<uint8_t>(base,0)); for(int i=0;i<base;i++){ int xi=D.reflection_gid(i); for(int j=0;j<base;j++){ int z=D.conj(xi,D.reflection_gid(j)); op[i][j]=(uint8_t)D.a_of(z);} }
 MonoBlocks mb=monodromy_blocks_Dn(n,base,powB,D); ConjClasses cc=conjugacy_classes(D);
 std::vector<KData> kd(n+1); for(int k=2;k<=n;k++) kd[k]=precompute_kdata(n,k,P);
 auto t0=std::chrono::high_resolution_clock::now(); auto qsh=precompute_qshuffle_selected(n,base,P,q_scalar,op,powB,{}); auto t1=std::chrono::high_resolution_clock::now();
 std::vector<std::atomic<long long>> rank_sum(n+2); for(int i=0;i<n+2;i++) rank_sum[i].store(0);
 std::vector<std::atomic<long long>> time_ns_sum(n+2); for(int i=0;i<n+2;i++) time_ns_sum[i].store(0);
 std::vector<Task> tasks; tasks.reserve((size_t)(n-1)*cc.classes.size());
 for(int k=2;k<=n;k++) for(int cid=0; cid<(int)cc.classes.size(); cid++){ int rep=cc.rep[cid]; if(!mb.words_by_g[rep].empty()) tasks.push_back({k,rep,(int)cc.classes[cid].size()}); }
 std::atomic<int> next(0);
 int worker_count = nested_blocks ? std::max(1, threads / 3) : threads;
 worker_count = std::max(1, std::min(worker_count, (int)tasks.size()));
 int inner_wiedemann_threads = nested_blocks ? 2 : std::max(1, threads / worker_count);
 auto worker=[&](){ while(true){ int idx=next.fetch_add(1,std::memory_order_relaxed); if(idx>=(int)tasks.size()) return; const Task& t=tasks[idx]; auto ts=std::chrono::high_resolution_clock::now(); auto mat=compile_block_differential_matrix(n,P,qsh,powB,mb,kd[t.k],t.rep_g); int local_inner=inner_wiedemann_threads; std::uint64_t seed=1469598103934665603ULL ^ (std::uint64_t)t.k*1315423911ULL ^ (std::uint64_t)t.rep_g*2654435761ULL; int rk=rank_rectangular_wiedemann(mat,P,repeats,local_inner,seed); rank_sum[t.k].fetch_add((long long)rk*(long long)t.class_size,std::memory_order_relaxed); auto te=std::chrono::high_resolution_clock::now(); long long dt_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(te-ts).count(); time_ns_sum[t.k].fetch_add(dt_ns,std::memory_order_relaxed); }};
 std::vector<std::thread> pool; pool.reserve(worker_count); for(int t=0;t<worker_count;t++) pool.emplace_back(worker); for(auto& th:pool) th.join();
 std::vector<__int128> rank_d(n+2,0); std::vector<double> time_d(n+2,0.0);
 for(int k=2;k<=n;k++){ rank_d[k]=(__int128)rank_sum[k].load(std::memory_order_relaxed); time_d[k]=(double)time_ns_sum[k].load(std::memory_order_relaxed)/1e9; std::cout<<"rank(d_"<<k<<")="<<(long long)rank_d[k]<<" time="<<time_d[k]<<"s\n"; }
 auto t2=std::chrono::high_resolution_clock::now();
 std::cout<<std::fixed<<std::setprecision(6)<<"qshuffle_pre="<<std::chrono::duration<double>(t1-t0).count()<<"s total_rank_time="<<std::chrono::duration<double>(t2-t1).count()<<"s\n";
 std::cout<<"\nH^j dims (j=0..n-1):\n"; for(int j=0;j<n;j++){ int k=n-j; __int128 dimBk=binom_i128(n-1,k-1)*(__int128)powB[n]; __int128 Hj=dimBk-rank_d[k]-rank_d[k+1]; std::cout<<"H^"<<j<<" = "; print_i128(Hj); std::cout<<"\n"; }
 return 0; }
