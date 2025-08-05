
template<>
EdgeKey<true>::EdgeKey(int v0, int v1)
{
	V[0] = v0;
	V[1] = v1;
}

template<>
EdgeKey<false>::EdgeKey(int v0, int v1)
{
	if (v0 < v1)
	{
		// v0 is minimum
		V[0] = v0;
		V[1] = v1;
	}
	else
	{
		// v1 is minimum
		V[0] = v1;
		V[1] = v0;
	}
}