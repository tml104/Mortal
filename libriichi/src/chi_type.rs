use crate::tile::Tile;

#[derive(Clone, Copy)]
pub enum ChiType { // 定义吃的是上中下的那种，例如：23吃1是Low，13吃2是Mid，23吃4是High
    Low,
    Mid,
    High,
}

impl ChiType {
    // consumed: 用于吃的2张牌， tile： 吃下的牌
    #[must_use]
    pub fn new(consumed: [Tile; 2], tile: Tile) -> Self { 
        let a = consumed[0].deaka().as_u8();
        let b = consumed[1].deaka().as_u8();
        let min = a.min(b);
        let max = a.max(b);
        let tile_id = tile.deaka().as_u8();
        if tile_id < min {
            Self::Low
        } else if tile_id < max {
            Self::Mid
        } else {
            Self::High
        }
    }
}
