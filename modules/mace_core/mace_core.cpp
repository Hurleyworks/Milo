#include "berserkpch.h"
#include "mace_core.h"

ItemID HasId::sId = 0;

namespace mace
{

#ifdef USE_OIIO
	#include "excludeFromBuild/imaging/CacheHandler.cpp"
#endif

} // namespace mace
