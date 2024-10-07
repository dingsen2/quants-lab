from hummingbot.strategy_v2.backtesting.controllers_backtesting.market_making_backtesting import MarketMakingBacktesting
from hummingbot.strategy_v2.controllers.controller_base import ControllerConfigBase
from pandas import Series
from typing import Dict, List, Optional, Union
from hummingbot.strategy_v2.backtesting.executor_simulator_base import ExecutorSimulation
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors import CloseType
from controllers.market_making.avellaneda_stoikov_backtesting_v2 import AvellanedaStoikovBacktestingV2ControllerConfig
from controllers.market_making.avellaneda_stoikov_backtesting_v2 import AvellanedaStoikovBacktestingV2Controller
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig


class ASV2MarketMakingBacktesting(MarketMakingBacktesting):
    async def update_processed_data(self, row: Series):
        # Reasoning: for this function, we need to update the processed_data based on the data we got from the previous row.
        # However, current backtesting_engine_base.py only applies the update_processed_data function once when the
        # backtesting starts. This is good for the case that all the data is calculated based on data that is retrieved
        # based on current timestamp.
        #
        # However, for the ASV2, we need to update the processed_data based on the data that is retrieved based on the
        # timestamp of the previous row.

        # TODO: For now, we can call the Update_processed_data() in the controller.
        # FIXME: The volatility is going to be the same for all the rows in this case as our candles are fixed.
        # Need to fix it later.
        # TODO: integrate update_processed_data() with row


        # This should be fine since the previous information is already stored in the processed_data.
        await self.controller.update_processed_data()


    async def run_backtesting(
        self,
        controller_config: AvellanedaStoikovBacktestingV2ControllerConfig,
        start: int,
        end: int,
        backtesting_resolution: str = "1m",
        trade_cost: float = 0.0006
    ):
        # For each timestamp, we need to:
            # 1. update the market data
            # 2. update the processed data
            # 3. determine the executor actions
            # 4. simulate the executor actions
        # Load historical candles
        # TODO: For here, we need to fetch all the data 30 intervals before so that all the data for calculating sigma can be fetched.
        self.backtesting_data_provider.update_backtesting_time(start, end)

        controller_class = controller_config.get_controller_class()
        self.controller = controller_class(
            config=controller_config,
            market_data_provider=self.backtesting_data_provider,
            actions_queue=None
        ) # dingsen: This line will run the __init__() function in the controller.
        self.backtesting_resolution = backtesting_resolution
        await self.initialize_backtesting_data_provider() # fetch candles data here.
        # await self.update_processed_data()
        self.as_bt_records = []
        executors_info = await self.simulate_execution(trade_cost=trade_cost)
        results = self.summarize_results(executors_info)
        return {
            "executors": executors_info,
            "results": results,
            "bt_records": self.as_bt_records,
        }
    
    async def simulate_execution(self, trade_cost: float) -> list:
        """
        Simulates market making strategy over historical data, considering trading costs.

        Args:
            trade_cost (float): The cost per trade.

        Returns:
            List[ExecutorInfo]: List of executor information objects detailing the simulation results.
        """
        candles_df = self.prepare_market_data()
        self.active_executor_simulations: List[ExecutorSimulation] = []
        self.stopped_executors_info: List[ExecutorInfo] = []
        for i, row in candles_df.iterrows():
            # TODO: we need to customize the update_market_data function so that we could use specific data range to calculate sigma.
            self.update_market_data(row) # for each row, update the price and timestamp to the controller.
            await self.update_processed_data(row) # based on previous row's information.
            cur_processed_data = self.controller.processed_data
            cur_parameters = self.controller.parameters
            combined_data = {
                'timestamp': row['timestamp'],
                **cur_processed_data,
                **cur_parameters
            }
            self.as_bt_records.append(combined_data)
            self.update_executors_info(row["timestamp"])
            for action in self.controller.determine_executor_actions():
                if isinstance(action, CreateExecutorAction):
                    executor_simulation = self.simulate_executor(action.executor_config, candles_df.loc[i:], trade_cost)
                    if executor_simulation.close_type != CloseType.FAILED:
                        self.manage_active_executors(executor_simulation)
                elif isinstance(action, StopExecutorAction):
                    self.handle_stop_action(action, row["timestamp"])

            # TODO: update the self.controller.processed_data here for the next row.

        return self.controller.executors_info
    
    def prepare_market_data(self):
        """
        Prepares market data by merging candle data with strategy features, filling missing values.

        Returns:
            pd.DataFrame: The prepared market data with necessary features.
        """
        backtesting_candles = self.controller.market_data_provider.get_candles_df(
            connector_name=self.controller.config.connector_name,
            trading_pair=self.controller.config.trading_pair,
            interval=self.backtesting_resolution
        ).add_suffix("_bt")

        # if "features" not in self.controller.processed_data:
        #     backtesting_candles["reference_price"] = backtesting_candles["close_bt"]
        #     backtesting_candles["spread_multiplier"] = 1
        #     backtesting_candles["signal"] = 0
        # else:
        #     backtesting_candles = pd.merge_asof(backtesting_candles, self.controller.processed_data["features"],
        #                                         left_on="timestamp_bt", right_on="timestamp",
        #                                         direction="backward")
        backtesting_candles["timestamp"] = backtesting_candles["timestamp_bt"]
        backtesting_candles["open"] = backtesting_candles["open_bt"]
        backtesting_candles["high"] = backtesting_candles["high_bt"]
        backtesting_candles["low"] = backtesting_candles["low_bt"]
        backtesting_candles["close"] = backtesting_candles["close_bt"]
        backtesting_candles["volume"] = backtesting_candles["volume_bt"]
        backtesting_candles.dropna(inplace=True)
        # self.controller.processed_data["features"] = backtesting_candles
        return backtesting_candles
    
    async def initialize_backtesting_data_provider(self):
        backtesting_config = CandlesConfig(
            connector=self.controller.config.connector_name,
            trading_pair=self.controller.config.trading_pair,
            interval=self.backtesting_resolution
        )
        await self.controller.market_data_provider.initialize_candles_feed(backtesting_config)
        for config in self.controller.config.candles_config:
            await self.controller.market_data_provider.initialize_candles_feed(config)