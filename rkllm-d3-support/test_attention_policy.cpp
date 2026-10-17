#include <cassert>
#include <functional>
#include <iostream>
#include "patches/d3_attention_policy.hpp"
using namespace d3_rkllm;
static void rejects(const std::function<void()>& fn) {
  bool threw=false;
  try { fn(); } catch (const std::exception&) { threw=true; }
  assert(threw);
}
int main() {
  const auto L=LayerKind::LinearAttention, F=LayerKind::FullAttention;
  const std::vector<LayerKind> types={L,L,F,L,L,F,L,L,F,F,F,F};
  assert(types.size()==12);
  assert(is_causal(L) && !is_causal(F));
  assert(select_layer(types,2)==F && select_layer(types,9)==F && select_layer(types,10)==F);
  rejects([&]{select_layer(types,12);});
  std::vector<uint8_t> keys(2*129,1);
  keys[0]=keys[1]=0; keys[129]=0;
  const auto mask=full_attention_mask(F,keys,2,129,0,16,129);
  assert(mask.size()==2*16*129);
  assert(mask[128]==1);                // Query 0 can read the FUTURE key in the next chunk.
  assert(mask[0]==0 && mask[1]==0);     // Left padding is masked.
  assert(mask[16*129+1]==1);            // Row 1 uses its own padding mask, not row 0's.
  for (size_t q=1;q<16;++q) assert(mask[q*129+128]==1);
  const auto tail=full_attention_mask(F,keys,2,129,128,1,129);
  assert(tail[2]==1 && tail[128]==1);
  rejects([&]{full_attention_mask(F,keys,2,129,0,16,16);});  // Incomplete chunk K/V.
  rejects([&]{full_attention_mask(F,keys,2,129,0,16,129,1);});
  rejects([&]{full_attention_mask(L,keys,2,129,0,16,129);});
  rejects([&]{full_attention_mask(F,keys,2,129,128,2,129);});
  rejects([&]{full_attention_mask(F,std::vector<uint8_t>(1),2,129,0,16,129);});
  std::cout << "PASS: original layout; future visibility across chunks; per-row padding; query bounds; incomplete K/V rejection; history rejection; linear scan boundary\n";
}
