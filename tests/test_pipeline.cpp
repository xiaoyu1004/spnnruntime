#include "pipeline.h"
#include "cpu.h"

static int testLoadParam()
{
    spnnruntime::pipeline::ptr pipe(new spnnruntime::pipeline);
    pipe->loadParam("/home/yunjian/projects/spnnruntime/models/NanJing.params");
    pipe->loadModel("/home/yunjian/projects/spnnruntime/models/NanJing.bin");
    // pipe->enableProfiler();

    std::cout << "pipeline start run!" << std::endl;
    pipe->start();

    spnnruntime::Extractor::ptr ex = pipe->getExtractor();

    spnnruntime::ToDataProc proc;
    std::uint32_t cnt = 0;
    while (cnt++ <= 160000)
    {
        ex->extract(&proc);
    }

    pipe->stop();

    return 0;
}

int main()
{
    return 0 || testLoadParam();
}