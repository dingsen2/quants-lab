from decimal import Decimal
from typing import List, Optional, Tuple, Union
from pydantic import Field, validator
from hummingbot.core.data_type.common import PriceType, TradeType
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
import numpy as np

from decimal import Decimal
from typing import List

from pydantic import Field
import pandas_ta as ta  # noqa: F401
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.market_making_controller_base import (
    MarketMakingControllerBase,
    MarketMakingControllerConfigBase,
)
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig, CandlesFactory


class AvellanedaStoikovBacktestingV2ControllerConfig(MarketMakingControllerConfigBase):
    """
    This class represents the base configuration for a market making controller.
    """
    controller_name = "avellaneda_stoikov_backtesting_v2"
    candles_config: List[CandlesConfig] = Field(default=[], client_data=ClientFieldData(
        prompt_on_new=False))  # dingsen: candles_config is configured using other ways.

    # dingsen: note here we are using our own spreads and amount so no need to use the basic ones.
    buy_spreads: List[float] = Field(default=[], client_data=ClientFieldData(prompt_on_new=False))
    sell_spreads: List[float] = Field(default=[], client_data=ClientFieldData(prompt_on_new=False))
    # buy_amounts_pct: Union[List[float], None] = Field(default=[], client_data=ClientFieldData(prompt_on_new=False))
    # sell_amounts_pct: Union[List[float], None] = Field(default=[], client_data=ClientFieldData(prompt_on_new=False))

    # dingsen: newly added fields
    target_base_quote_ratio: Optional[Decimal] = Field(
        default=Decimal("0.5"), gt=0,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt=lambda mi: "Enter target inventory (as a decimal, e.g., 0.5 for 50% base-50% quote): ",
            prompt_on_new=True))
    inventory_risk_aversion: Optional[Decimal] = Field(
        default=Decimal("0.5"), gt=0,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt=lambda mi: "Enter your inventory risk aversion index (as a decimal, e.g., 0.5 for 50%): ",
            prompt_on_new=True))

    start_base_balance: Optional[Decimal] = Field(
        default=Decimal("10.0"), gt=0,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt=lambda mi: "Enter the starting base_balance for backtesting (as a decimal, e.g., 10.0): ",
            prompt_on_new=True))

    start_quote_balance: Optional[Decimal] = Field(
        default=Decimal("1000.0"), gt=0,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt=lambda mi: "Enter the starting quote_balance for backtesting (as a decimal, e.g., 1000.0): ",
            prompt_on_new=True))

    min_order_price: Optional[Decimal] = Field(
        default=Decimal("39.86"), gt=0,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt=lambda mi: "Enter the minimum order price(as a decimal, e.g. 39.86 for ETH, you can find it on the "
                              "connector's website): ",
            prompt_on_new=True))

    gamma: Optional[Decimal] = Field(
        default=Decimal("2.0"), gt=0,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt=lambda mi: "Enter the risk factor value(\u03B3) (as a decimal, e.g., 2.0): ",
            prompt_on_new=True))

    kappa: Optional[Decimal] = Field(
        default=Decimal("7.25"), gt=0,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt=lambda mi: "Enter the order book liquidity value(\u03BA) (as a decimal, e.g., 7.25): ",
            prompt_on_new=True))

    # candles parameters
    candles_connector: str = Field(
        default="binance",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the name of the exchange to fetch candles data from (e.g., binance): "))

    candles_trading_pair: str = Field(
        default=None,
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda
                mi: "Enter the trading pair for the candles data, leave empty to use the same trading pair as the connector: ", )
    )

    interval: str = Field(
        default="30m",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the interval of the candles data (e.g., 1m, 5m, 1h, 1d): "))

    max_records: int = Field(
        default=1000,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the maximum number of candles to fetch (e.g., 1000): ",
            prompt_on_new=True))

    natr_length: int = Field(
        default=30,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the candles length(How many candles you want to use to measure natr): ",
            prompt_on_new=True))

    @validator("candles_trading_pair", pre=True, always=True)
    def set_candles_trading_pair(cls, v, values):
        if v is None or v == "":
            return values.get("trading_pair")
        return v


class AvellanedaStoikovBacktestingV2Controller(MarketMakingControllerBase):
    """
    dingsen: This class is the controller for the Avellaneda-Stoikov market making strategy.
    """

    def __init__(self, config: AvellanedaStoikovBacktestingV2ControllerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        #FIXME: dingsen: This assumes that one controller(one self instance) only has one connector.
        # because we only specify one connector name for each controller instance.
        # self.connector = self.market_data_provider.get_connector(self.config.connector_name)
        self.base, self.quote = self.config.trading_pair.split('-')
        # Here you can use for example the LastTrade price to use in your strategy
        self.price_source = PriceType.MidPrice
        self.candles = None
        self.max_records = self.config.max_records

        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]

        self.parameters = {
            "q": Decimal(0),
            "sigma": Decimal(0),
            "gamma_max": Decimal(0),
            "gamma": Decimal(0),
            "kappa": Decimal(0),
            "time_left": Decimal(1)
        }
        self.processed_data = {
            "reference_price": Decimal(0),
            "reservation_price": Decimal(0),
            "optimal_spread": Decimal(0),
            "inventory": Decimal(0),
            "base_balance": self.config.start_base_balance,
            "quote_balance": self.config.start_quote_balance,
        }
        self.level_amount = 1

    async def update_processed_data(self):
        """
        Update the processed data for the controller. This method should be reimplemented to modify the reference price
        and spread multiplier based on the market data. By default, it will update the reference price as mid price and
        the spread multiplier as 1.
        TODO: dingsen This function should also update all the parameters.
        TODO: dingsen: this function should update the reference price and spread
        """

        self.candles = self.market_data_provider.get_candles_df(connector_name=self.config.candles_connector,
                                                                trading_pair=self.config.candles_trading_pair,
                                                                interval=self.config.interval,
                                                                max_records=self.config.max_records)
        self.processed_data["reference_price"] = Decimal(
            self.market_data_provider.get_price_by_type(self.config.connector_name,
                                                        self.config.trading_pair, PriceType.MidPrice))
        # dingsen: update parameters.
        self.parameters["q"] = self.calculate_inventory_deviation()
        self.parameters["sigma"] = self.calculate_market_volatility()
        self.parameters["gamma"] = self.calculate_risk_factor()
        self.parameters["kappa"] = self.calculate_order_book_liquidity()
        # print("self.parameters:", self.parameters)

        # dingsen: update processed data.
        self.processed_data["reservation_price"] = Decimal(self.calculate_reservation_price())
        self.processed_data["optimal_spread"] = Decimal(self.calculate_optimal_spread())
        self.processed_data["optimal_one_side_spread"] = (self.processed_data["optimal_spread"] / 2
                                                          / self.processed_data["reservation_price"])

        self.processed_data["buy_spread_pct"] = self.processed_data["optimal_one_side_spread"]
        self.processed_data["sell_spread_pct"] = self.processed_data["buy_spread_pct"]

        # update total_filled_amount_quote
        # Note that this function will only be called when the refresh time is reached. So after this function is
        # called, all the executors will be refreshed.
        for executor_info in self.executors_info:
            if not executor_info.is_trading:
                continue
            if executor_info.config.side == TradeType.BUY: # Now it's buying base asset
                self.processed_data["base_balance"] += executor_info.filled_amount_quote
                self.processed_data["quote_balance"] -= executor_info.config.amount
            else: # Now it's selling base asset
                self.processed_data["base_balance"] -= executor_info.filled_amount_quote
                self.processed_data["quote_balance"] += executor_info.config.amount

    def quantize_order_amount(self, amount: Decimal):
        """
        dingsen: This function should quantize the order amount based on the min order size and min amount increment
        """
        order_size_quantum = self.processed_data["reference_price"] / self.config.min_order_price
        return (amount // order_size_quantum) * order_size_quantum

    def calculate_target_inventory(self):
        price = self.processed_data["reference_price"]
        base_balance = self.processed_data["base_balance"]
        quote_balance = self.processed_data["quote_balance"]
        # Base asset value in quote asset prices
        base_value = base_balance * price
        # Total inventory value in quote asset prices
        inventory_value = base_value + quote_balance
        # Target base asset value in quote asset prices
        target_inventory_value = inventory_value * self.config.target_base_quote_ratio
        # Target base asset amount
        target_inventory_amount = target_inventory_value / price
        return self.quantize_order_amount(Decimal(str(target_inventory_amount)))

    def calculate_current_inventory(self):
        price = self.processed_data["reference_price"]
        base_balance = self.processed_data["base_balance"]
        quote_balance = self.processed_data["quote_balance"]
        # Base asset value in quote asset prices
        base_value = base_balance * price
        # Total inventory value in quote asset prices
        inventory_value_quote = base_value + quote_balance
        # Total inventory value in base asset prices
        inventory_value_base = inventory_value_quote / price
        return inventory_value_base

    def calculate_inventory_deviation(self):
        """
        author: dingsen
        This function calculates the inventory deviation parameter $q$.
        trader is short of base -> should buy more -> raising the reserved price -> q < 0
        trader is long of base -> should sell more -> decreasing the reserved price -> q > 0
        The paper suggests a complex way to deduce q that I don't understand lol. I'll use what Michael
        used in inventory shift video as q here.

        TODO: Understand what's going on in the paper and calculated q based on that.
        """
        q_target = Decimal(str(self.calculate_target_inventory()))
        current_inventory = Decimal(str(self.calculate_current_inventory()))
        base_balance = self.processed_data["base_balance"]
        q = (base_balance - q_target) / current_inventory
        # print("q", q)
        return q

    def calculate_market_volatility(self):
        """
        author: dingsen
        This function calculate the market volatility parameter $sigma$.
        In the simple version of A&S impl, this function just fetches the NATR value of last order
        Not used in this script
        """
        # dingsen: here we are fetching the ATR as a signal of volatility
        candles_df = self.candles
        candles_df.ta.natr(length=self.config.natr_length, scalar=1, append=True)
        sigma = Decimal(candles_df[f"NATR_{self.config.natr_length}"].iloc[-1])  # set gamma to be NATR
        return sigma

    def calculate_risk_factor(self):
        """
        author: dingsen
        For now gamma is selected by users.
        """
        return self.config.gamma

    def calculate_order_book_liquidity(self):
        """
        author: dingsen
        This function calculate the order book liquidity parameter $kappa$.
        higher kappa -> more liquid -> smaller spread to be competitive to hit orders.
        lower kappa -> less liquid -> larger spread to make more profits.

        note that here we let the users choose $kappa$ themselves.
        """
        return self.config.kappa

    def calculate_reservation_price(self):
        """
        author: dingsen
        This function calculate the reservation price.
        return: the reservation price
        formula: r=s-q*gamma*sigma^2*(T-t)
        """
        # self.vamp_price = self.calculate_vamp()
        # Use original mid price for now.(reference price)
        # TODO: include vamp as the orig_price
        reservation_price = (self.processed_data["reference_price"] - self.parameters["q"] * self.parameters["gamma"] *
                             self.parameters["sigma"] * self.parameters["sigma"] * self.parameters["time_left"])
        return reservation_price

    def calculate_optimal_spread(self):
        """
        author: dingsen
        This function calculate the optimal spread.
        formula: gamma * sigma^2 * time_left + 2/gamma * ln(1 + self.gamma / self.kappa)
        return: the optimal spread
        """
        optimal_spread = self.parameters["gamma"] * self.parameters["sigma"] ** 2 * self.parameters["time_left"]
        optimal_spread += 2 * Decimal(1 + self.parameters["gamma"] / self.parameters["kappa"]).ln() / self.parameters[
            "gamma"]
        return optimal_spread

    def get_executor_config(self, level_id: str, price: Decimal, amount: Decimal):
        trade_type = self.get_trade_type_from_level_id(level_id)
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            level_id=level_id,
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            entry_price=price,
            amount=amount,
            triple_barrier_config=self.config.triple_barrier_config,
            leverage=self.config.leverage,
            side=trade_type,
        )

    def get_price_and_amount(self, level_id) -> Tuple[Decimal, Decimal]:
        """
        dingsen: this function should return the price and amount for the executor
        """
        trade_type = self.get_trade_type_from_level_id(level_id)
        order_amount = self.config.total_amount_quote
        reservation_price = self.processed_data["reservation_price"]
        # dingsen: in this case, we're placing orders on both sides with the same spreads.
        spread_in_pct = self.processed_data["buy_spread_pct"]

        side_multiplier = Decimal("-1") if trade_type == TradeType.BUY else Decimal("1")
        order_price = reservation_price * (1 + side_multiplier * spread_in_pct)
        return (order_price,
                Decimal(order_amount) / order_price)  # dingsen: divided by order_price to get the amount in base asset

    def get_not_active_levels_ids(self, active_levels_ids: List[str]) -> List[str]:
        """
        Get the levels to execute based on the current state of the controller.
        """
        buy_ids_missing = [self.get_level_id_from_side(TradeType.BUY, level) for level in range(self.level_amount)
                           if self.get_level_id_from_side(TradeType.BUY, level) not in active_levels_ids]
        sell_ids_missing = [self.get_level_id_from_side(TradeType.SELL, level) for level in range(self.level_amount)
                            if self.get_level_id_from_side(TradeType.SELL, level) not in active_levels_ids]
        return buy_ids_missing + sell_ids_missing
