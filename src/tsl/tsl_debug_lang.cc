#include <iomanip>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/te/schedule.h>
#include <tvm/tsl/tsl_debug_lang.h>

using namespace std;

namespace tvm {
namespace te {
void TslPrintDecomposeCtx(const StageNode* stage) {
  auto ctx=stage->decompose_ctx;
  cout << "IterType:";
  for (auto& stack:ctx) {
    cout<<setw(20)<<stack.iter_type;
  }
  cout<<endl;
  cout<<"stacks:"<<endl;
  cout<<setw(6)<<"Factor"<<setw(20)<<"Path"<<endl;
  size_t layer=0;
  bool out=false;
  do {
    out=false;
    for (size_t i = 0; i < ctx.size(); i++) {
      if(layer<ctx[i].size()) {
        auto& entry=ctx[i][layer];
        cout<<setw(6)<<entry.factor<<setw(40)<<entry.pathivar<<flush;
        out=true;
      }
    }
    layer++;
    cout<<endl;
  }while(out);
  
}

}
}