/// reduction table
///        [a0|
///        |a1|
///        |a2]
///        [a3|
///        |a4|
///        |a5]
///        [a6|
///        |a7]
/// ---------- +
///    [b0|[a8|
///    |b1||a9|
///    |b2]|aA]
/// ---------- +
/// c0 [b3| aB
///    |b4]
/// ---------- +
/// c0  b5  aB
/// c1
kernel void step(const global ulong* field, global ulong* new_field, const ulong width)
{
    const size_t x = get_global_id(0);
    const size_t y = get_global_id(1);
    const size_t i = x + y * width;

    //__local shared_field[64];

    //event_t e = async_work_group_copy(shared_field, field + i, 1, 0);
    //wait_group_events(1, &e);

    ulong result = field[i];

    // top: left mid right
    const ulong a0 = (field[i - width] >> 1) | (field[i - width - 1] << 63);
    const ulong a1 = field[i - width];
    const ulong a2 = (field[i - width] << 1) | (field[i - width + 1] >> 63);

    // middle: left right
    const ulong a3 = (result >> 1) | (field[i - 1] << 63);
    const ulong a4 = (result << 1) | (field[i + 1] >> 63);

    // bottom: left mid right
    const ulong a5 = (field[i + width] >> 1) | (field[i + width - 1] << 63);
    const ulong a6 = field[i + width];
    const ulong a7 = (field[i + width] << 1) | (field[i + width + 1] >> 63);

    // stage 0
    const ulong ta0 = a0 ^ a1;
    const ulong a8 = ta0 ^ a2;
    const ulong b0 = (a0 & a1) | (ta0 & a2);

    const ulong ta3 = a3 ^ a4;
    const ulong a9 = ta3 ^ a5;
    const ulong b1 = (a3 & a4) | (ta3 & a5);

    const ulong aA = a6 ^ a7;
    const ulong b2 = a6 & a7;

    // stage 1
    const ulong ta8 = a8 ^ a9;
    const ulong aB = ta8 ^ aA;
    const ulong b3 = (a8 & a9) | (ta8 & aA);

    const ulong tb0 = b0 ^ b1;
    const ulong b4 = tb0 ^ b2;
    const ulong c0 = (b0 & b1) | (tb0 & b2);

    // stage 2
    const ulong b5 = b3 ^ b4;
    const ulong c1 = b3 & b4;

    result |= aB;
    result &= b5;
    result &= ~c0;
    result &= ~c1;

    new_field[i] = result;
}