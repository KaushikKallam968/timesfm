import uuid
from datetime import datetime, timezone


class OrderManager:
    def __init__(self, client, db, risk_manager, mock_mode=True):
        self.client = client
        self.db = db
        self.risk_manager = risk_manager
        self.mock_mode = mock_mode
        self._orders = {}

    def place_order(self, token_id, side, size, price):
        if not self.risk_manager.can_trade(size):
            return {"id": None, "status": "rejected", "reason": "risk_limit", "filled_size": 0}

        order_id = str(uuid.uuid4())

        if self.mock_mode:
            self.risk_manager.add_position()
            self.db.log_trade(
                market_id=token_id,
                side=side,
                price=price,
                size=size,
                edge=0.0,
                truth_source="mock",
                truth_probability=0.0,
            )
            order = {
                "id": order_id,
                "token_id": token_id,
                "side": side,
                "size": size,
                "price": price,
                "status": "filled",
                "filled_size": size,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._orders[order_id] = order
            return order

        raise NotImplementedError("Live mode requires py-clob-client")

    def cancel_order(self, order_id):
        if order_id not in self._orders:
            return {"id": order_id, "status": "not_found"}
        order = self._orders[order_id]
        if order["status"] == "filled":
            return {"id": order_id, "status": "already_filled"}
        order["status"] = "cancelled"
        return {"id": order_id, "status": "cancelled"}

    def get_open_orders(self):
        return [o for o in self._orders.values() if o["status"] == "pending"]

    def check_settlements(self):
        if self.mock_mode:
            return []
        raise NotImplementedError("Live mode requires py-clob-client")
